//! Multi-database creation benchmark measuring creation rate (DBs/sec),
//! latency percentiles, retained RSS/VSZ per database, open file descriptors,
//! and teardown time across Turso (WAL, MVCC, PostgreSQL) and SQLite.
//!
//! Short runs mostly measure thread start-up. Pick `--databases` so a run
//! lasts about a second, and compare the medians of several runs.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use mimalloc::MiMalloc;
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;
use tempfile::TempDir;
use turso_core::{
    ConnectOpts, Connection, Database, DatabaseOpts, MemoryIO, OpenFlags, OpenOptions, PlatformIO,
    SessionRole, SyncMode, IO,
};
use turso_pg::PgConnection;
use turso_sqlite::SqliteDialect;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const INIT_TABLE_SQL: &str = "CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    amount INTEGER NOT NULL,
    note TEXT
)";
const INIT_INSERT_SQL: &str =
    "INSERT INTO items (id, sku, amount, note) VALUES (1, 'sku-1', 10, 'initial')";

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Engine {
    TursoWal,
    TursoMvcc,
    TursoPg,
    Sqlite,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum StorageMode {
    Memory,
    File,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Durability {
    Off,
    Full,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Stage {
    Open,
    Connect,
    Ddl,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
    Csv,
}

#[derive(Clone, Copy, Parser, Debug)]
#[command(about = "Benchmark creating many databases across threads")]
struct Args {
    #[arg(long, value_enum, default_value_t = Engine::TursoWal)]
    engine: Engine,

    #[arg(long, value_enum, default_value_t = StorageMode::Memory)]
    storage: StorageMode,

    #[arg(long, value_enum, default_value_t = Durability::Off)]
    sync: Durability,

    #[arg(long, value_enum, default_value_t = Stage::Ddl)]
    stage: Stage,

    #[arg(short = 'n', long, default_value_t = 500)]
    databases: usize,

    #[arg(short = 't', long, default_value_t = 1)]
    threads: usize,

    /// Keep all created databases and connections alive until RSS and open FDs are sampled.
    #[arg(long)]
    retain: bool,

    #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
    format: OutputFormat,
}

#[derive(Serialize)]
struct Report {
    engine: Engine,
    storage: StorageMode,
    sync: Durability,
    stage: Stage,
    databases: usize,
    threads: usize,
    retain: bool,
    elapsed_ms: f64,
    dbs_per_sec: f64,
    drop_ms: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
    rss_before_kb: f64,
    rss_after_kb: f64,
    rss_delta_kb: f64,
    rss_per_db_kb: f64,
    vsz_delta_kb: f64,
    vsz_per_db_kb: f64,
    fds_before: Option<usize>,
    fds_after: Option<usize>,
    fds_delta: Option<isize>,
    fds_per_db: Option<f64>,
    vmas_delta: Option<isize>,
}

#[expect(
    dead_code,
    reason = "instances are retained in memory until drop is measured"
)]
enum RetainedInstance {
    Turso {
        db: Arc<Database>,
        conn: Option<Arc<Connection>>,
    },
    TursoPg {
        db: Arc<Database>,
        conn: Option<PgConnection>,
    },
    Sqlite {
        conn: rusqlite::Connection,
    },
}

#[derive(Clone, Copy)]
struct ProcessResources {
    rss_bytes: usize,
    vsz_bytes: usize,
    open_fds: Option<usize>,
    vmas: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.databases > 0, "--databases must be greater than 0");
    anyhow::ensure!(args.threads > 0, "--threads must be greater than 0");

    let report = execute_benchmark(&args)?;
    print_report(&report, args.format)?;
    Ok(())
}

fn execute_benchmark(args: &Args) -> Result<Report> {
    let temp_dir = match args.storage {
        StorageMode::Memory => None,
        StorageMode::File => Some(TempDir::new().context("create temp dir")?),
    };
    let shared_io: Option<Arc<dyn IO>> = match args.storage {
        StorageMode::Memory => None,
        StorageMode::File => match args.engine {
            Engine::Sqlite => None,
            Engine::TursoWal | Engine::TursoMvcc | Engine::TursoPg => {
                #[allow(clippy::arc_with_non_send_sync)]
                let io: Arc<dyn IO> = Arc::new(PlatformIO::new().context("init PlatformIO")?);
                Some(io)
            }
        },
    };

    warm_up_once(args, shared_io.as_ref())?;

    let worker_count = args.threads.min(args.databases);
    let next_id = Arc::new(AtomicUsize::new(0));
    let start_barrier = Arc::new(Barrier::new(worker_count + 1));
    let created_barrier = Arc::new(Barrier::new(worker_count + 1));
    let release_barrier = Arc::new(Barrier::new(worker_count + 1));

    let before = sample_resources()?;

    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let next_id = Arc::clone(&next_id);
        let start_barrier = Arc::clone(&start_barrier);
        let created_barrier = Arc::clone(&created_barrier);
        let release_barrier = Arc::clone(&release_barrier);
        let dir_path = temp_dir.as_ref().map(|d| d.path().to_path_buf());
        let shared_io = shared_io.clone();
        let worker_args = *args;

        handles.push(std::thread::spawn(move || -> Result<Vec<u64>> {
            let per_worker = worker_args.databases.div_ceil(worker_count);
            let mut retained = if worker_args.retain {
                Vec::with_capacity(per_worker)
            } else {
                Vec::new()
            };
            let mut latencies_ns = Vec::with_capacity(per_worker);

            start_barrier.wait();
            let created = loop {
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                if id >= worker_args.databases {
                    break Ok(());
                }
                let t0 = Instant::now();
                let instance =
                    match create_one(&worker_args, id, dir_path.as_deref(), shared_io.as_ref()) {
                        Ok(instance) => instance,
                        Err(error) => break Err(error),
                    };
                latencies_ns.push(t0.elapsed().as_nanos() as u64);
                if worker_args.retain {
                    retained.push(instance);
                } else {
                    drop(instance);
                }
            };
            // A failed worker still waits on both barriers: the main thread
            // blocks on each one until every worker arrives.
            created_barrier.wait();
            release_barrier.wait();
            drop(retained);
            created.map(|()| latencies_ns)
        }));
    }

    start_barrier.wait();
    let create_start = Instant::now();
    created_barrier.wait();
    let create_elapsed = create_start.elapsed();

    let after = sample_resources()?;

    let drop_start = Instant::now();
    release_barrier.wait();
    let mut all_latencies_ns = Vec::with_capacity(args.databases);
    for handle in handles {
        let worker_latencies = handle
            .join()
            .map_err(|_| anyhow::anyhow!("worker thread panicked"))??;
        all_latencies_ns.extend(worker_latencies);
    }
    let drop_elapsed = drop_start.elapsed();

    all_latencies_ns.sort_unstable();
    let elapsed_secs = create_elapsed.as_secs_f64();
    let dbs_f = args.databases as f64;
    let rss_delta_kb = (after.rss_bytes as f64 - before.rss_bytes as f64) / 1024.0;
    let vsz_delta_kb = (after.vsz_bytes as f64 - before.vsz_bytes as f64) / 1024.0;
    let fds_delta = before
        .open_fds
        .zip(after.open_fds)
        .map(|(b, a)| a as isize - b as isize);
    let vmas_delta = before
        .vmas
        .zip(after.vmas)
        .map(|(b, a)| a as isize - b as isize);

    Ok(Report {
        engine: args.engine,
        storage: args.storage,
        sync: args.sync,
        stage: args.stage,
        databases: args.databases,
        threads: worker_count,
        retain: args.retain,
        elapsed_ms: elapsed_secs * 1000.0,
        dbs_per_sec: dbs_f / elapsed_secs,
        drop_ms: drop_elapsed.as_secs_f64() * 1000.0,
        p50_us: percentile_us(&all_latencies_ns, 0.50),
        p95_us: percentile_us(&all_latencies_ns, 0.95),
        p99_us: percentile_us(&all_latencies_ns, 0.99),
        max_us: all_latencies_ns.last().copied().unwrap_or(0) as f64 / 1000.0,
        rss_before_kb: before.rss_bytes as f64 / 1024.0,
        rss_after_kb: after.rss_bytes as f64 / 1024.0,
        rss_delta_kb,
        rss_per_db_kb: rss_delta_kb / dbs_f,
        vsz_delta_kb,
        vsz_per_db_kb: vsz_delta_kb / dbs_f,
        fds_before: before.open_fds,
        fds_after: after.open_fds,
        fds_delta,
        fds_per_db: fds_delta.map(|d| d as f64 / dbs_f),
        vmas_delta,
    })
}

fn warm_up_once(args: &Args, shared_io: Option<&Arc<dyn IO>>) -> Result<()> {
    let warmup_dir = TempDir::new().context("create warmup dir")?;
    drop(create_one(args, 0, Some(warmup_dir.path()), shared_io)?);
    Ok(())
}

fn create_one(
    args: &Args,
    id: usize,
    dir: Option<&Path>,
    shared_io: Option<&Arc<dyn IO>>,
) -> Result<RetainedInstance> {
    let path_buf;
    let path = match args.storage {
        StorageMode::Memory => ":memory:",
        StorageMode::File => {
            let base = dir.context("file storage requires a directory")?;
            path_buf = base.join(format!("db_{id}.db"));
            path_buf
                .to_str()
                .context("database path must be valid UTF-8")?
        }
    };

    match args.engine {
        Engine::TursoWal | Engine::TursoMvcc => {
            let io = turso_io(args.storage, shared_io)?;
            let enable_mvcc = matches!(args.engine, Engine::TursoMvcc);
            let opts = OpenOptions::new(Arc::new(SqliteDialect))
                .flags(OpenFlags::Create)
                .db_opts(DatabaseOpts::new().with_mvcc(enable_mvcc));
            let db = Database::open(io, path, opts)?;
            let conn = match args.stage {
                Stage::Open => None,
                Stage::Connect | Stage::Ddl => {
                    let conn = db.connect(ConnectOpts::new(SessionRole::Superuser))?;
                    match args.sync {
                        Durability::Off => conn.set_sync_mode(SyncMode::Off),
                        Durability::Full => conn.set_sync_mode(SyncMode::Full),
                    }
                    if matches!(args.stage, Stage::Ddl) {
                        conn.execute(INIT_TABLE_SQL)?;
                        conn.execute(INIT_INSERT_SQL)?;
                    }
                    Some(conn)
                }
            };
            Ok(RetainedInstance::Turso { db, conn })
        }
        Engine::TursoPg => {
            let io = turso_io(args.storage, shared_io)?;
            let db =
                turso_pg::open_database_with_io(io, path, OpenFlags::Create, DatabaseOpts::new())?;
            let conn = match args.stage {
                Stage::Open => None,
                Stage::Connect | Stage::Ddl => {
                    let conn = PgConnection::connect(&db)?;
                    match args.sync {
                        Durability::Off => conn.inner().set_sync_mode(SyncMode::Off),
                        Durability::Full => conn.inner().set_sync_mode(SyncMode::Full),
                    }
                    if matches!(args.stage, Stage::Ddl) {
                        conn.execute(INIT_TABLE_SQL)?;
                        conn.execute(INIT_INSERT_SQL)?;
                    }
                    Some(conn)
                }
            };
            Ok(RetainedInstance::TursoPg { db, conn })
        }
        Engine::Sqlite => {
            let conn = match args.storage {
                StorageMode::Memory => rusqlite::Connection::open_in_memory()?,
                StorageMode::File => {
                    let c = rusqlite::Connection::open(path)?;
                    c.pragma_update(None, "journal_mode", "WAL")?;
                    match args.sync {
                        Durability::Off => c.pragma_update(None, "synchronous", "OFF")?,
                        Durability::Full => c.pragma_update(None, "synchronous", "FULL")?,
                    }
                    c
                }
            };
            match args.stage {
                Stage::Open | Stage::Connect => {}
                Stage::Ddl => {
                    conn.execute(INIT_TABLE_SQL, ())?;
                    conn.execute(INIT_INSERT_SQL, ())?;
                }
            }
            Ok(RetainedInstance::Sqlite { conn })
        }
    }
}

fn turso_io(storage: StorageMode, shared_io: Option<&Arc<dyn IO>>) -> Result<Arc<dyn IO>> {
    match storage {
        StorageMode::Memory => {
            #[allow(clippy::arc_with_non_send_sync)]
            let mem_io: Arc<dyn IO> = Arc::new(MemoryIO::new());
            Ok(mem_io)
        }
        StorageMode::File => Ok(Arc::clone(shared_io.context("missing shared PlatformIO")?)),
    }
}

fn sample_resources() -> Result<ProcessResources> {
    let mem = memory_stats::memory_stats().context("query process memory stats")?;
    Ok(ProcessResources {
        rss_bytes: mem.physical_mem,
        vsz_bytes: mem.virtual_mem,
        open_fds: count_open_fds(),
        vmas: count_linux_vmas(),
    })
}

fn count_open_fds() -> Option<usize> {
    for dir in ["/proc/self/fd", "/dev/fd"] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let count = entries.count();
            // Subtract 1 for the directory descriptor opened by `read_dir` itself.
            return Some(count.saturating_sub(1));
        }
    }
    None
}

fn count_linux_vmas() -> Option<usize> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    Some(maps.lines().count())
}

fn percentile_us(sorted_ns: &[u64], quantile: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ns.len() - 1) as f64 * quantile).round() as usize;
    sorted_ns[idx] as f64 / 1000.0
}

fn print_report(r: &Report, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(r)?);
        }
        OutputFormat::Csv => {
            println!(
                "engine,storage,sync,stage,databases,threads,retain,dbs_per_sec,elapsed_ms,drop_ms,p50_us,p95_us,p99_us,rss_per_db_kb,vsz_per_db_kb,fds_per_db"
            );
            println!(
                "{:?},{:?},{:?},{:?},{},{},{},{:.1},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2}",
                r.engine,
                r.storage,
                r.sync,
                r.stage,
                r.databases,
                r.threads,
                r.retain,
                r.dbs_per_sec,
                r.elapsed_ms,
                r.drop_ms,
                r.p50_us,
                r.p95_us,
                r.p99_us,
                r.rss_per_db_kb,
                r.vsz_per_db_kb,
                r.fds_per_db.unwrap_or(0.0),
            );
        }
        OutputFormat::Human => {
            println!(
                "=== DB CREATION ({:?} | {:?} | sync={:?} | stage={:?}) ===",
                r.engine, r.storage, r.sync, r.stage
            );
            println!(
                "Databases: {}  Threads: {}  Retain: {}",
                r.databases, r.threads, r.retain
            );
            println!(
                "Throughput: {:.1} DBs/sec  (create: {:.2} ms, drop: {:.2} ms)",
                r.dbs_per_sec, r.elapsed_ms, r.drop_ms
            );
            println!(
                "Latency:    p50={:.1} µs  p95={:.1} µs  p99={:.1} µs  max={:.1} µs",
                r.p50_us, r.p95_us, r.p99_us, r.max_us
            );
            println!(
                "RSS delta:  {:+.1} KB ({:+.1} KB / DB)  |  VSZ delta: {:+.1} KB ({:+.1} KB / DB)",
                r.rss_delta_kb, r.rss_per_db_kb, r.vsz_delta_kb, r.vsz_per_db_kb
            );
            if let (Some(delta), Some(per_db)) = (r.fds_delta, r.fds_per_db) {
                println!("Open FDs:   {delta:+} ({per_db:.2} FDs / DB)");
            }
            if let Some(vmas) = r.vmas_delta {
                println!("VMA delta:  {vmas:+}");
            }
        }
    }
    Ok(())
}
