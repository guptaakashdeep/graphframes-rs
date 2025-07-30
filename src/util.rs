use crate::GraphFrame;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::error::Result;
use datafusion::prelude::{CsvReadOptions, SessionContext};
use std::collections::HashMap;
use std::io::Result as ioResult;
use std::{env, fs};

// Gets the basepath of the dataset based on if it's benchmark runs or test runs
/// # Arguments
///
/// * `benchmark_run`: true for benchmark runs to read data from bench/data dir, false for tests to read data from testing/data.
pub fn _get_dataset_base_path(benchmark_run: bool) -> Result<String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir_name = if benchmark_run { "benches" } else { "testing" };
    let base_path = format!("{}/{}/data/ldbc", manifest_dir, dir_name);
    Ok(base_path)
}

/// Creates a GraphFrame from an LDBC-style dataset.
///
/// The dataset is expected to be located in the `benches/data/` directory.
///
/// # Arguments
///
/// * `dataset`: The name of the dataset directory (e.g., "test-pr-directed").
/// * `benchmark_run`: true for benchmark runs to read data from bench/data dir, false for tests to read data from testing/data.
pub async fn create_ldbc_test_graph(dataset: &str, benchmark_run: bool) -> Result<GraphFrame> {
    let ctx = SessionContext::new();

    let edge_schema = Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]);
    let vertices_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);

    let ds_base_path = _get_dataset_base_path(benchmark_run)?;

    let edges_path = format!("{}/{}/{}.e.csv", ds_base_path, dataset, dataset);
    let vertices_path = format!("{}/{}/{}.v.csv", ds_base_path, dataset, dataset);

    println!("{}", edges_path);
    println!("{}", vertices_path);

    let edges = ctx
        .read_csv(
            &edges_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&edge_schema),
        )
        .await?;

    let vertices = ctx
        .read_csv(
            &vertices_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&vertices_schema),
        )
        .await?;
    Ok(GraphFrame { vertices, edges })
}

// Reads the ldbc dataset properties file and converts it into a HashMap
pub fn parse_ldbc_properties_file(
    dataset: &str,
    benchmark_run: bool,
) -> ioResult<HashMap<String, String>> {
    let prop_fp = format!(
        "{}/{}/{}.properties",
        _get_dataset_base_path(benchmark_run)?,
        dataset,
        dataset
    );
    let content = fs::read_to_string(prop_fp)?;
    let mut properties_map: HashMap<String, String> = HashMap::new();

    for line in content.lines() {
        let trimmed_line = line.trim();

        if trimmed_line.is_empty() || trimmed_line.starts_with("#") {
            continue;
        }
        if let Some((key, value)) = trimmed_line.split_once("=") {
            properties_map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    Ok(properties_map)
}
