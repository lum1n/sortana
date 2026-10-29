//! Evaluate the rubric against labelled tasks using live Jev.
//!
//! `TYPESAFE_API_KEY=... cargo run --example eval [-- path/to/cases.jsonl]`
//!
//! Each line of the file is `{"text": "...", "tier": "simple|medium|hard|very_hard"}`,
//! optionally with `"effort": "low|medium|high"`. Prints tier accuracy, the
//! off-by-two-or-more rate (gate: under 5%), latency percentiles and a
//! confusion matrix; then how often Jev answered the effort question, effort
//! accuracy on the labelled cases with its own confusion matrix, and how many
//! tasks were flagged as needing vision or tools. Exits non-zero when the
//! tier gate fails. Effort is reported, not gated.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use sortana::{Classifier, Effort, OnError, Tier};

const GATE_OFF_BY_TWO: f64 = 0.05;

#[tokio::main]
async fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "eval/cases.jsonl".into());
    let key = std::env::var("TYPESAFE_API_KEY").expect("TYPESAFE_API_KEY is not set");
    let classifier = Classifier::builder()
        .jev(key)
        .on_error(OnError::Fail)
        .build()
        .expect("build classifier");

    let cases: Vec<(String, Tier, Option<Effort>)> = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {path}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            let v: serde_json::Value =
                serde_json::from_str(l).unwrap_or_else(|e| panic!("line {}: {e}", i + 1));
            let text = v["text"].as_str().expect("text").to_owned();
            let tier = v["tier"].as_str().expect("tier").parse().expect("tier");
            let effort = v["effort"].as_str().map(|e| e.parse().expect("effort"));
            (text, tier, effort)
        })
        .collect();

    // Sequential so latency reflects single requests.
    let mut confusion = [[0usize; 4]; 4];
    let mut latencies = Vec::new();
    let (mut correct, mut off2, mut raw_correct, mut errors) = (0, 0, 0, 0);
    let mut effort_confusion = [[0usize; 3]; 3];
    let (mut answered, mut labelled, mut effort_correct) = (0, 0, 0);
    let (mut vision, mut tools) = (0, 0);
    for (text, want, want_effort) in &cases {
        let start = Instant::now();
        let r = classifier.classify(text).await;
        latencies.push(start.elapsed());
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                errors += 1;
                eprintln!("error: {e}: {text}");
                continue;
            }
        };
        confusion[want.index()][r.tier.index()] += 1;
        correct += usize::from(r.tier == *want);
        raw_correct += usize::from(r.raw_tier == *want);
        let dist = want.index().abs_diff(r.tier.index());
        if dist >= 2 {
            off2 += 1;
        }
        if r.tier != *want {
            eprintln!(
                "miss: want {want}, got {} (raw {}, conf {:.2}): {text}",
                r.tier, r.raw_tier, r.confidence
            );
        }
        vision += usize::from(r.needs.vision);
        tools += usize::from(r.needs.tools);
        let Some(got) = r.effort else { continue };
        answered += 1;
        let Some(want) = want_effort else { continue };
        labelled += 1;
        effort_confusion[*want as usize][got as usize] += 1;
        effort_correct += usize::from(got == *want);
        if got != *want {
            eprintln!("effort miss: want {want}, got {got}: {text}");
        }
    }

    let n = cases.len().max(1) as f64;
    let off2_rate = off2 as f64 / n;
    latencies.sort();
    println!("cases            {}", cases.len());
    println!("errors           {errors}");
    println!("accuracy         {:.1}%", 100.0 * correct as f64 / n);
    println!(
        "raw accuracy     {:.1}%  (before confidence bump)",
        100.0 * raw_correct as f64 / n
    );
    println!(
        "off by >=2 tiers {:.1}%  (gate < {:.0}%)",
        100.0 * off2_rate,
        100.0 * GATE_OFF_BY_TWO
    );
    println!("latency p50      {:?}", percentile(&latencies, 0.50));
    println!("latency p95      {:?}", percentile(&latencies, 0.95));
    println!("rubric           {}", sortana::rubric::RUBRIC_VERSION);
    println!("\nconfusion (rows = labelled, cols = predicted)");
    println!(
        "{:>10} {:>7} {:>7} {:>7} {:>9}",
        "", "simple", "medium", "hard", "very_hard"
    );
    for t in Tier::ALL {
        let row = confusion[t.index()];
        println!(
            "{:>10} {:>7} {:>7} {:>7} {:>9}",
            t.as_str(),
            row[0],
            row[1],
            row[2],
            row[3]
        );
    }

    let classified = (cases.len() - errors).max(1) as f64;
    println!("\neffort answered  {answered}/{}", cases.len() - errors);
    if labelled > 0 {
        println!(
            "effort accuracy  {:.1}%  ({labelled} labelled)",
            100.0 * effort_correct as f64 / labelled as f64
        );
        println!("\neffort confusion (rows = labelled, cols = predicted)");
        println!("{:>10} {:>7} {:>7} {:>7}", "", "low", "medium", "high");
        for e in Effort::ALL {
            let row = effort_confusion[e as usize];
            println!(
                "{:>10} {:>7} {:>7} {:>7}",
                e.as_str(),
                row[0],
                row[1],
                row[2]
            );
        }
    }
    println!(
        "\nneeds vision     {vision} ({:.0}%)",
        100.0 * vision as f64 / classified
    );
    println!(
        "needs tools      {tools} ({:.0}%)",
        100.0 * tools as f64 / classified
    );

    if off2_rate < GATE_OFF_BY_TWO && errors == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}
