//! `sortana [OPTIONS] [TASK...]`: classify a task given as arguments or on
//! stdin, and print its tier and model. Reads the Jev key from `TYPESAFE_API_KEY`.

use std::io::Read;
use std::process::ExitCode;

use sortana::{Host, Lineup, ModelMap, Tier};

const USAGE: &str = "\
usage: sortana [OPTIONS] [TASK...]
Reads the task from stdin when no TASK is given.

options:
  -l, --lineup ID   model lineup to route to (default: anthropic)
      --host ID     resolve model ids for this host (default: the lineup's own API)
      --opencode    print ids in opencode's host/model format (host defaults to opencode)
      --json        print the full result as JSON
      --list [HOST] list lineups, with tier coverage on HOST if given, then exit
      --hosts       list known hosts, then exit
  -h, --help        show this help";

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
    let mut opencode = false;
    let mut lineup = "anthropic".to_owned();
    let mut host: Option<String> = None;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--opencode" => opencode = true,
            "-l" | "--lineup" => lineup = args.next().ok_or("--lineup needs a value")?,
            "--host" => host = Some(args.next().ok_or("--host needs a value")?),
            "--list" => {
                let host = args.next_if(|a| !a.starts_with('-'));
                return list_lineups(host.as_deref());
            }
            "--hosts" => {
                for h in Host::all() {
                    println!("{:<28} {}", h.id, h.name);
                }
                return Ok(());
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => words.push(arg),
        }
    }

    let models = match (opencode, host) {
        (true, host) => ModelMap::opencode(&lineup, host.as_deref().unwrap_or("opencode"))?,
        (false, Some(host)) => ModelMap::preset_on(&lineup, &host)?,
        (false, None) => ModelMap::preset(&lineup)?,
    };

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
        .models(models)
        .on_error(sortana::OnError::Fail)
        .build()?;
    let classifier = sortana::blocking::Classifier::new(classifier)?;
    let r = classifier.classify(&text)?;

    if json {
        println!("{}", serde_json::to_string(&r)?);
    } else {
        let p = r.probabilities.map(|p| format!("{p:.2}")).join(" ");
        let bumped = if r.bumped {
            format!(" (bumped from {})", r.raw_tier)
        } else {
            String::new()
        };
        println!(
            "{}\t{}\tconfidence={:.2}\tp=[{p}]{bumped}",
            r.tier, r.model, r.confidence
        );
    }
    Ok(())
}

/// Print each lineup's models. With a host, print the resolved ids and mark
/// tiers the host lacks (filled from the nearest tier) with `*`.
fn list_lineups(host: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(h) = host
        && Host::get(h).is_none()
    {
        return Err(format!("unknown host {h:?}; see --hosts").into());
    }
    println!("catalogue: models.dev, {}", sortana::CATALOG_DATE);
    for l in Lineup::all() {
        let (ids, cov) = match host {
            None => (ModelMap::preset(l.id)?, [true; 4]),
            Some(h) => match ModelMap::preset_on(l.id, h) {
                Ok(m) => (m, l.coverage(h)),
                Err(_) => continue,
            },
        };
        let cols: Vec<String> = Tier::ALL
            .iter()
            .map(|&t| {
                let mark = if cov[t.index()] { "" } else { "*" };
                format!("{}{mark}", ids.get(t))
            })
            .collect();
        println!("{:<15} {}", l.id, cols.join("  "));
    }
    if host.is_some() {
        println!("* not on this host; uses the nearest tier's model, preferring stronger");
    }
    Ok(())
}
