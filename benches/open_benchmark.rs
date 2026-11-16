//! Database creation and reopen benchmarks (time + heap allocations).
//!
//! Covers the full single-database creation funnel (`open` -> `open + connect`
//! -> `open + connect + DDL`) across in-memory and file-backed storage, WAL and
//! MVCC modes, and a local `rusqlite` comparison.
//!
//! Run with:
//!   cargo bench -p turso_benches --bench open_benchmark

use divan::{black_box, AllocProfiler, Bencher};
use mimalloc::MiMalloc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tempfile::TempDir;
use turso_core::{
    ConnectOpts, Connection, Database, DatabaseOpts, MemoryIO, OpenFlags, OpenOptions, PlatformIO,
    SessionRole, SyncMode, IO,
};
use turso_sqlite::SqliteDialect;

#[global_allocator]
static ALLOC: AllocProfiler<MiMalloc> = AllocProfiler::new(MiMalloc);

const INIT_TABLE_SQL: &str = "CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    amount INTEGER NOT NULL,
    note TEXT
)";
const INIT_INSERT_SQL: &str =
    "INSERT INTO items (id, sku, amount, note) VALUES (1, 'sku-1', 10, 'initial')";

#[cfg(not(feature = "codspeed"))]
fn main() {
    divan::Divan::default()
        .sample_size(1)
        .sample_count(150)
        .config_with_args()
        .main();
}

#[cfg(feature = "codspeed")]
fn main() {
    divan::main();
}

#[turso_macros::divan_bench_single_call]
fn memory_wal_1_open(bencher: Bencher) {
    let dialect = Arc::new(SqliteDialect);

    bencher
        .with_inputs(|| {
            #[allow(clippy::arc_with_non_send_sync)]
            let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
            (io, OpenOptions::new(dialect.clone()))
        })
        .bench_local_values(|(io, opts)| {
            Database::open(black_box(io), black_box(":memory:"), opts).unwrap()
        });
}

#[turso_macros::divan_bench_single_call]
fn memory_wal_2_open_connect(bencher: Bencher) {
    let dialect = Arc::new(SqliteDialect);

    bencher
        .with_inputs(|| {
            #[allow(clippy::arc_with_non_send_sync)]
            let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
            (io, OpenOptions::new(dialect.clone()))
        })
        .bench_local_values(|(io, opts)| -> (Arc<Database>, Arc<Connection>) {
            let db = Database::open(black_box(io), black_box(":memory:"), opts).unwrap();
            let conn = db
                .connect(ConnectOpts::new(SessionRole::Superuser))
                .unwrap();
            (db, conn)
        });
}

#[turso_macros::divan_bench_single_call]
fn memory_wal_3_open_connect_ddl(bencher: Bencher) {
    let dialect = Arc::new(SqliteDialect);

    bencher
        .with_inputs(|| {
            #[allow(clippy::arc_with_non_send_sync)]
            let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
            (io, OpenOptions::new(dialect.clone()))
        })
        .bench_local_values(|(io, opts)| -> (Arc<Database>, Arc<Connection>) {
            let db = Database::open(black_box(io), black_box(":memory:"), opts).unwrap();
            let conn = db
                .connect(ConnectOpts::new(SessionRole::Superuser))
                .unwrap();
            conn.execute(INIT_TABLE_SQL).unwrap();
            conn.execute(INIT_INSERT_SQL).unwrap();
            (db, conn)
        });
}

#[turso_macros::divan_bench_single_call]
fn memory_mvcc_3_open_connect_ddl(bencher: Bencher) {
    let dialect = Arc::new(SqliteDialect);

    bencher
        .with_inputs(|| {
            #[allow(clippy::arc_with_non_send_sync)]
            let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
            (
                io,
                OpenOptions::new(dialect.clone()).db_opts(DatabaseOpts::new().with_mvcc(true)),
            )
        })
        .bench_local_values(|(io, opts)| -> (Arc<Database>, Arc<Connection>) {
            let db = Database::open(black_box(io), black_box(":memory:"), opts).unwrap();
            let conn = db
                .connect(ConnectOpts::new(SessionRole::Superuser))
                .unwrap();
            conn.execute(INIT_TABLE_SQL).unwrap();
            conn.execute(INIT_INSERT_SQL).unwrap();
            (db, conn)
        });
}

#[turso_macros::divan_bench_single_call]
fn memory_custom_types_1_open(bencher: Bencher) {
    let dialect = Arc::new(SqliteDialect);

    bencher
        .with_inputs(|| {
            #[allow(clippy::arc_with_non_send_sync)]
            let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
            (
                io,
                OpenOptions::new(dialect.clone())
                    .db_opts(DatabaseOpts::new().with_custom_types(true)),
            )
        })
        .bench_local_values(|(io, opts)| {
            Database::open(black_box(io), black_box(":memory:"), opts).unwrap()
        });
}

#[cfg(not(feature = "codspeed"))]
#[divan::bench]
fn memory_sqlite_3_open_connect_ddl(bencher: Bencher) {
    bencher.bench_local(|| {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute(INIT_TABLE_SQL, ()).unwrap();
        conn.execute(INIT_INSERT_SQL, ()).unwrap();
        conn
    });
}

#[turso_macros::divan_bench_single_call]
fn file_wal_1_create_empty(bencher: Bencher) {
    let temp_dir = TempDir::new().unwrap();
    #[allow(clippy::arc_with_non_send_sync)]
    let io: Arc<dyn IO> = Arc::new(PlatformIO::new().unwrap());
    let dialect = Arc::new(SqliteDialect);
    let next_id = AtomicU64::new(0);

    bencher
        .with_inputs(|| {
            let path = next_db_path(&temp_dir, "empty", &next_id);
            (
                path,
                OpenOptions::new(dialect.clone()).flags(OpenFlags::Create),
            )
        })
        .bench_local_values(|(path, opts)| {
            Database::open(black_box(io.clone()), black_box(&path), opts).unwrap()
        });
}

#[turso_macros::divan_bench_single_call]
fn file_wal_2_create_connect(bencher: Bencher) {
    let temp_dir = TempDir::new().unwrap();
    #[allow(clippy::arc_with_non_send_sync)]
    let io: Arc<dyn IO> = Arc::new(PlatformIO::new().unwrap());
    let dialect = Arc::new(SqliteDialect);
    let next_id = AtomicU64::new(0);

    bencher
        .with_inputs(|| {
            let path = next_db_path(&temp_dir, "connect", &next_id);
            (
                path,
                OpenOptions::new(dialect.clone()).flags(OpenFlags::Create),
            )
        })
        .bench_local_values(|(path, opts)| -> (Arc<Database>, Arc<Connection>) {
            let db = Database::open(black_box(io.clone()), black_box(&path), opts).unwrap();
            let conn = db
                .connect(ConnectOpts::new(SessionRole::Superuser))
                .unwrap();
            (db, conn)
        });
}

#[turso_macros::divan_bench_single_call]
fn file_wal_3_create_connect_ddl_sync_off(bencher: Bencher) {
    let temp_dir = TempDir::new().unwrap();
    #[allow(clippy::arc_with_non_send_sync)]
    let io: Arc<dyn IO> = Arc::new(PlatformIO::new().unwrap());
    let dialect = Arc::new(SqliteDialect);
    let next_id = AtomicU64::new(0);

    bencher
        .with_inputs(|| {
            let path = next_db_path(&temp_dir, "ddl", &next_id);
            (
                path,
                OpenOptions::new(dialect.clone()).flags(OpenFlags::Create),
            )
        })
        .bench_local_values(|(path, opts)| -> (Arc<Database>, Arc<Connection>) {
            let db = Database::open(black_box(io.clone()), black_box(&path), opts).unwrap();
            let conn = db
                .connect(ConnectOpts::new(SessionRole::Superuser))
                .unwrap();
            conn.set_sync_mode(SyncMode::Off);
            conn.execute(INIT_TABLE_SQL).unwrap();
            conn.execute(INIT_INSERT_SQL).unwrap();
            (db, conn)
        });
}

#[cfg(not(feature = "codspeed"))]
#[divan::bench]
fn file_sqlite_3_create_connect_ddl_sync_off(bencher: Bencher) {
    let temp_dir = TempDir::new().unwrap();
    let next_id = AtomicU64::new(0);

    bencher
        .with_inputs(|| next_db_path(&temp_dir, "sqlite_ddl", &next_id))
        .bench_local_values(|path| {
            let conn = rusqlite::Connection::open(black_box(&path)).unwrap();
            conn.pragma_update(None, "journal_mode", "WAL").unwrap();
            conn.pragma_update(None, "synchronous", "OFF").unwrap();
            conn.execute(INIT_TABLE_SQL, ()).unwrap();
            conn.execute(INIT_INSERT_SQL, ()).unwrap();
            conn
        });
}

#[turso_macros::divan_bench_single_call]
fn memory_wal_reopen_20_tables(bencher: Bencher) {
    #[allow(clippy::arc_with_non_send_sync)]
    let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
    let dialect = Arc::new(SqliteDialect);
    let path = ":memory:seeded_20";
    {
        let db = Database::open(io.clone(), path, OpenOptions::new(dialect.clone())).unwrap();
        seed_schema(&db, 20);
        let conn = db
            .connect(ConnectOpts::new(SessionRole::Superuser))
            .unwrap();
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
    }

    bencher
        .with_inputs(|| OpenOptions::new(dialect.clone()))
        .bench_local_values(|opts| {
            Database::open(black_box(io.clone()), black_box(path), opts).unwrap()
        });
}

#[turso_macros::divan_bench_single_call]
fn file_wal_reopen_20_tables(bencher: Bencher) {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir
        .path()
        .join("seeded_20.db")
        .to_string_lossy()
        .into_owned();
    #[allow(clippy::arc_with_non_send_sync)]
    let io: Arc<dyn IO> = Arc::new(PlatformIO::new().unwrap());
    let dialect = Arc::new(SqliteDialect);

    {
        let db = Database::open(
            io.clone(),
            &path,
            OpenOptions::new(dialect.clone()).flags(OpenFlags::Create),
        )
        .unwrap();
        seed_schema(&db, 20);
        let conn = db
            .connect(ConnectOpts::new(SessionRole::Superuser))
            .unwrap();
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
    }

    bencher
        .with_inputs(|| OpenOptions::new(dialect.clone()).flags(OpenFlags::Create))
        .bench_local_values(|opts| {
            Database::open(black_box(io.clone()), black_box(&path), opts).unwrap()
        });
}

fn next_db_path(temp_dir: &TempDir, prefix: &str, next_id: &AtomicU64) -> String {
    let id = next_id.fetch_add(1, Ordering::Relaxed);
    temp_dir
        .path()
        .join(format!("{prefix}_{id}.db"))
        .to_string_lossy()
        .into_owned()
}

fn seed_schema(db: &Arc<Database>, table_count: usize) {
    let conn = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    conn.set_sync_mode(SyncMode::Off);
    for i in 0..table_count {
        conn.execute(format!(
            "CREATE TABLE table_{i} (
                id INTEGER PRIMARY KEY,
                sku TEXT NOT NULL,
                amount INTEGER NOT NULL,
                note TEXT
            )"
        ))
        .unwrap();
        conn.execute(format!(
            "CREATE INDEX idx_table_{i}_sku ON table_{i}(sku, amount)"
        ))
        .unwrap();
    }
}
