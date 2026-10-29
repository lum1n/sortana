//! `sortana [OPTIONS] [TASK...]`: classify a task given as arguments or on
//! stdin, and print its tier and model. Reads the Jev key from `TYPESAFE_API_KEY`.

use std::io::Read;
use std::process::ExitCode;

use sortana::{Context, Decision, Host, Lineup, ModelMap, Tier};

const USAGE: &str = "\
usage: sortana [OPTIONS] [TASK...]
Reads the task from stdin when no TASK is given.

options:
  -l, --lineup ID   model lineup to route to (default: anthropic)
      --host ID     resolve model ids for this host (default: the lineup's own API)
      --opencode    print ids in opencode's host/model format (host defaults to opencode)
      --min-tier T  never route below tier T (simple, medium, hard, very_hard)
      --max-tier T  never route above tier T
      --cost [P]    pick the tier with the lowest expected cost; P is the miss
                    penalty (default 1)
      --files N     files attached or in scope
      --images N    images attached (the model must accept images)
      --tokens N    prompt size in tokens (the model's input limit must fit it)
      --turns N     conversation turns so far
      --tools       tools will be offered; --no-tools: they will not
      --notes TEXT  other context for the classifier
      --no-signals  only ask for the tier, not effort, vision or tools
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
    let mut ctx = Context::default();
    let mut builder = sortana::Classifier::builder().on_error(sortana::OnError::Fail);
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--opencode" => opencode = true,
            "-l" | "--lineup" => lineup = args.next().ok_or("--lineup needs a value")?,
            "--min-tier" => builder = builder.min_tier(value(&mut args, &arg)?),
            "--max-tier" => builder = builder.max_tier(value(&mut args, &arg)?),
            "--cost" => {
                let miss_penalty = match args.next_if(|a| a.parse::<f32>().is_ok()) {
                    Some(p) => p.parse()?,
                    None => 1.0,
                };
                builder = builder.decision(Decision::Cost { miss_penalty });
            }
            "--files" => ctx.files = Some(value(&mut args, &arg)?),
            "--images" => ctx.images = value(&mut args, &arg)?,
            "--tokens" => ctx.prompt_tokens = Some(value(&mut args, &arg)?),
            "--turns" => ctx.turns = Some(value(&mut args, &arg)?),
            "--tools" => ctx.tools = Some(true),
            "--no-tools" => ctx.tools = Some(false),
            "--notes" => ctx.notes = Some(args.next().ok_or("--notes needs a value")?),
            "--no-signals" => builder = builder.ask_effort(false).ask_needs(false),
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
    let classifier = builder.jev(key).models(models).build()?;
    let classifier = sortana::blocking::Classifier::new(classifier)?;
    let r = classifier.classify_with(&text, &ctx)?;

    if json {
        println!("{}", serde_json::to_string(&r)?);
    } else {
        let p = r.probabilities.map(|p| format!("{p:.2}")).join(" ");
        let mut line = format!(
            "{}\t{}\tconfidence={:.2}\tp=[{p}]",
            r.tier, r.model, r.confidence
        );
        if let Some(e) = r.effort {
            line += &format!("\teffort={e}");
            if let Some(v) = r.reasoning_effort {
                line += &format!(" ({v})");
            }
        }
        let needs: Vec<&str> = [(r.needs.vision, "vision"), (r.needs.tools, "tools")]
            .into_iter()
            .filter_map(|(on, name)| on.then_some(name))
            .collect();
        if !needs.is_empty() {
            line += &format!("\tneeds={}", needs.join(","));
        }
        let moves: Vec<String> = [
            (r.bumped, "bumped".to_owned()),
            (r.clamped, "clamped".to_owned()),
            (r.refit, "refit for needs".to_owned()),
            (
                r.fits == Some(false),
                "model lacks needs, none stronger has them".to_owned(),
            ),
        ]
        .into_iter()
        .filter_map(|(on, s)| on.then_some(s))
        .collect();
        if r.tier != r.raw_tier || !moves.is_empty() {
            line += &format!("\t(from {}; {})", r.raw_tier, moves.join(", "));
        }
        println!("{line}");
    }
    Ok(())
}

/// Parse the value after `flag`.
fn value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    let v = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
    v.parse().map_err(|e| format!("{flag} {v:?}: {e}").into())
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
