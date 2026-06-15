use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
enum Position {
    Goalkeeper,
    Defender,
    Midfielder,
    Forward,
}

struct PlayerRow {
    position: Position,
    goals: f64,
    assists: f64,
    shots_on_target: f64,
    expected_goals_xg: f64,
    expected_assists_xa: f64,
    key_passes: f64,
    successful_passes: f64,
    successful_dribbles: f64,
    tackles: f64,
    interceptions: f64,
    recoveries: f64,
    fouls_committed: f64,
    yellow_cards: f64,
    red_cards: f64,
    saves: f64,
    clean_sheet: f64,
    goals_conceded: f64,
    clearances: f64,
    blocks: f64,
    aerial_duels_won: f64,
    pass_accuracy: f64,
    possession_impact: f64,
    creativity_score: f64,
    offensive_contribution: f64,
    clutch_performance_score: f64, 
}

fn impact_score(row: &PlayerRow) -> f64 {
    let mut score =
        row.goals * 4.0
        + row.assists * 3.0
        + row.shots_on_target * 0.7
        + row.expected_goals_xg * 1.5
        + row.expected_assists_xa * 1.2
        + row.key_passes * 0.5
        + row.successful_passes * 0.01
        + row.successful_dribbles * 0.4
        + row.tackles * 0.3
        + row.interceptions * 0.3
        + row.recoveries * 0.1
        - row.fouls_committed * 0.4
        - row.yellow_cards * 1.0
        - row.red_cards * 3.0;

    match row.position {
        Position::Goalkeeper => {
            score +=
                row.saves * 0.8
                + row.clean_sheet * 3.0
                - row.goals_conceded * 0.7;
        }

        Position::Defender => {
            score +=
                row.clearances * 0.25
                + row.blocks * 0.4
                + row.aerial_duels_won * 0.2;
        }

        Position::Midfielder => {
            score +=
                row.pass_accuracy * 0.03
                + row.possession_impact * 0.5
                + row.creativity_score * 0.4;
        }

        Position::Forward => {
            score +=
                row.offensive_contribution * 0.6
                + row.clutch_performance_score * 0.4;
        }
    }

    score
}
#[pyfunction]
fn hello() -> String {
    "hello from rust".to_string()
}

#[pyfunction]
fn compute_scores(
    positions: Vec<u8>,
    goals: Vec<f64>,
    assists: Vec<f64>,
    shots_on_target: Vec<f64>,
    expected_goals_xg: Vec<f64>,
    expected_assists_xa: Vec<f64>,
    key_passes: Vec<f64>,
    successful_passes: Vec<f64>,
    successful_dribbles: Vec<f64>,
    tackles: Vec<f64>,
    interceptions: Vec<f64>,
    recoveries: Vec<f64>,
    fouls_committed: Vec<f64>,
    yellow_cards: Vec<f64>,
    red_cards: Vec<f64>,
    saves: Vec<f64>,
    clean_sheet: Vec<f64>,
    goals_conceded: Vec<f64>,
    clearances: Vec<f64>,
    blocks: Vec<f64>,
    aerial_duels_won: Vec<f64>,
    pass_accuracy: Vec<f64>,
    possession_impact: Vec<f64>,
    creativity_score: Vec<f64>,
    offensive_contribution: Vec<f64>,
    clutch_performance_score: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let n = goals.len();

    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let position = match positions[i] {
            0 => Position::Goalkeeper,
            1 => Position::Defender,
            2 => Position::Midfielder,
            3 => Position::Forward,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Invalid position: {}", positions[i]),
                ));
            }
        };

        let row = PlayerRow {
            position,
            goals: goals[i],
            assists: assists[i],
            shots_on_target: shots_on_target[i],
            expected_goals_xg: expected_goals_xg[i],
            expected_assists_xa: expected_assists_xa[i],
            key_passes: key_passes[i],
            successful_passes: successful_passes[i],
            successful_dribbles: successful_dribbles[i],
            tackles: tackles[i],
            interceptions: interceptions[i],
            recoveries: recoveries[i],
            fouls_committed: fouls_committed[i],
            yellow_cards: yellow_cards[i],
            red_cards: red_cards[i],
            saves: saves[i],
            clean_sheet: clean_sheet[i],
            goals_conceded: goals_conceded[i],
            clearances: clearances[i],
            blocks: blocks[i],
            aerial_duels_won: aerial_duels_won[i],
            pass_accuracy: pass_accuracy[i],
            possession_impact: possession_impact[i],
            creativity_score: creativity_score[i],
            offensive_contribution: offensive_contribution[i],
            clutch_performance_score: clutch_performance_score[i],
        };

        scores.push(impact_score(&row));
    }

    Ok(scores)
}


#[pymodule]
fn football_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_scores, m)?)?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}