import fastf1
import pandas as pd
import os

# FastF1 caches session data locally so it doesn't re-download on every run
# Create a cache folder in your project
os.makedirs("cache/fastf1", exist_ok=True)
fastf1.Cache.enable_cache("cache/fastf1")

def get_tyre_stint_summary(year: int, round_number: int, driver: str) -> pd.DataFrame:
    """
    Loads a race session and returns a per-stint summary:
    compound, stint number, laps on tyre, avg lap time, and pace degradation.
    """
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps.pick_drivers(driver).pick_quicklaps()

    # Group by stint and compound
    stints = (
        laps.groupby(["Stint", "Compound"])
        .agg(
            lap_count=("LapNumber", "count"),
            avg_lap_time_s=("LapTime", lambda x: x.dt.total_seconds().mean()),
            min_lap_time_s=("LapTime", lambda x: x.dt.total_seconds().min()),
            max_lap_time_s=("LapTime", lambda x: x.dt.total_seconds().max()),
        )
        .reset_index()
    )

    # Pace degradation: difference between last and first lap time within stint
    def calc_degradation(group):
        times = group["LapTime"].dt.total_seconds()
        if len(times) < 2:
            return 0.0
        return round(times.iloc[-1] - times.iloc[0], 3)

    deg_per_stint = (
        laps.groupby(["Stint", "Compound"])
        .apply(calc_degradation)
        .reset_index()
        .rename(columns={0: "degradation_s"})
    )

    result = stints.merge(deg_per_stint, on=["Stint", "Compound"])
    return result


def get_race_context(year: int, round_number: int, driver: str, current_lap: int) -> dict:
    """
    Returns a structured race context dict that Node A will reason over.
    Includes current tyre info and stint summary up to the current lap.
    """
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps.pick_drivers(driver).pick_quicklaps()

    # Get laps up to current lap
    laps_so_far = laps[laps["LapNumber"] <= current_lap]

    if laps_so_far.empty:
        return {"error": "No lap data found up to specified lap"}

    # Current stint info
    current_lap_data = laps_so_far.iloc[-1]
    current_compound = current_lap_data["Compound"]
    current_stint = current_lap_data["Stint"]
    laps_on_current_tyre = int(laps_so_far[laps_so_far["Stint"] == current_stint]["LapNumber"].count())

    # Full stint summary
    stint_summary = get_tyre_stint_summary(year, round_number, driver)

    return {
        "driver": driver,
        "current_lap": current_lap,
        "total_laps": int(session.event["RoundNumber"]),  
        "current_compound": current_compound,
        "laps_on_current_tyre": laps_on_current_tyre,
        "current_stint_number": int(current_stint),
        "stint_summary": stint_summary.to_dict(orient="records"),
    }


# --- Quick test ---
if __name__ == "__main__":
    print("Loading 2023 Bahrain GP data for VER...")
    print("(First run will download and cache — takes ~30 seconds)\n")

    context = get_race_context(
        year=2023,
        round_number=1,   # Bahrain GP
        driver="VER",
        current_lap=40
    )

    print(f"Driver:               {context['driver']}")
    print(f"Current lap:          {context['current_lap']}")
    print(f"Current compound:     {context['current_compound']}")
    print(f"Laps on tyre:         {context['laps_on_current_tyre']}")
    print(f"Current stint:        {context['current_stint_number']}")
    print(f"\nStint summary:")
    for stint in context["stint_summary"]:
        print(f"  Stint {stint['Stint']} | {stint['Compound']:<8} | "
              f"{stint['lap_count']} laps | "
              f"avg {stint['avg_lap_time_s']:.3f}s | "
              f"deg {stint['degradation_s']:+.3f}s")