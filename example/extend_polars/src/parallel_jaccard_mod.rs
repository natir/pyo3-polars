use polars::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use rayon::prelude::*;

/// Create `n` splits so that we can slice a polars data structure
/// and process the chunks in parallel
fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect()
    }
}

fn compute_jaccard_similarity(sa: &Series, sb: &Series) -> PolarsResult<Series> {
    let sa = sa.list()?;
    let sb = sb.list()?;

    let ca = sa
        .into_iter()
        .zip(sb.into_iter())
        .map(|(a, b)| {
            match (a, b) {
                (Some(a), Some(b)) => {
                    // unpack as i64 series
                    let a = a.i64()?;
                    let b = b.i64()?;

                    // convert to hashsets over Option<i64>
                    let s1 = a.into_iter().collect::<PlHashSet<_>>();
                    let s2 = b.into_iter().collect::<PlHashSet<_>>();

                    // count the number of intersections
                    let s3_len = s1.intersection(&s2).count();
                    // return similarity
                    Ok(Some(s3_len as f64 / (s1.len() + s2.len() - s3_len) as f64))
                }
                _ => Ok(None),
            }
        })
        .collect::<PolarsResult<Float64Chunked>>()?;
    Ok(ca.into_series())
}

pub(super) fn parallel_jaccard(df: DataFrame, col_a: &str, col_b: &str) -> PolarsResult<DataFrame> {
    let offsets = split_offsets(df.height(), rayon::current_num_threads());

    let dfs = offsets
        .par_iter()
        .map(|(offset, len)| {
            let sub_df = df.slice(*offset as i64, *len);
            let a = sub_df.column(col_a)?;
            let b = sub_df.column(col_b)?;

            let out = compute_jaccard_similarity(a, b)?;
            df!(
                "jaccard" => out
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    accumulate_dataframes_vertical(dfs)
}

pub(super) fn polars_sum(mut lf: LazyFrame) -> PolarsResult<LazyFrame> {
    lf = lf.select([as_struct(&[col("list_a"), col("list_b")]).map(
        |s| {
            let ca = s.struct_()?;

            let series_a = ca.field_by_name("list_a")?;
            let series_b = ca.field_by_name("list_b")?;

            let chunked_a = series_a.u64()?;
            let chunked_b = series_b.u64()?;

            let out: UInt64Chunked = chunked_a
                .into_iter()
                .zip(chunked_b.into_iter())
                .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                })
                .collect();

            Ok(Some(out.into_series()))
        },
        GetOutput::from_type(DataType::UInt64),
    )]);

    Ok(lf)
}
