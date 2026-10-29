//! `sortana [--json] [TASK...]`: classify a task given as arguments or on stdin.
//! Reads the Jev key from `TYPESAFE_API_KEY`.

use std::io::Read;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sortana: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut json = false;
    let mut words = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--json" => json = true,
            "-h" | "--help" => {
                println!("usage: sortana [--json] [TASK...]   (reads stdin when no TASK is given)");
                return Ok(());
            }
            _ => words.push(arg),
        }
    }
    let text = if words.is_empty() {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        words.join(" ")
    };

    let key = std::env::var("TYPESAFE_API_KEY").map_err(|_| "TYPESAFE_API_KEY is not set")?;
    let classifier = sortana::Classifier::builder()
        .jev(key)
        .on_error(sortana::OnError::Fail)
        .build()?;
    let classifier = sortana::blocking::Classifier::new(classifier)?;
    let r = classifier.classify(&text)?;
    let model = classifier.inner().model_for(r.tier);

    if json {
        let mut v = serde_json::to_value(&r)?;
        v["model"] = model.into();
        println!("{v}");
    } else {
        let p = r.probabilities.map(|p| format!("{p:.2}")).join(" ");
        let bumped = if r.bumped {
            format!(" (bumped from {})", r.raw_tier)
        } else {
            String::new()
        };
        println!(
            "{}\t{model}\tconfidence={:.2}\tp=[{p}]{bumped}",
            r.tier, r.confidence
        );
    }
    Ok(())
}
