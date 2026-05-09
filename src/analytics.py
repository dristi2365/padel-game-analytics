import pandas as pd
import json

def save_results(results_list):
    """
    Save detection results to CSV and JSON formats.
    Also prints a summary of shot analytics to the console.
    """

    #Covert list of detections to a pandas DataFrame(like a spreadsheet)
    df = pd.DataFrame(results_list)

    #Save to CSV
    df.to_csv("output/shots.csv", index=False)

    #Save to JSON - structured format as required by the assignment
    with open("output/shots.json", "w") as f:
        json.dump(results_list, f, indent=4)

    #Print overall shot type counts
    print("\n--- Shot Analytics ---")
    print(df["shot_type"].value_counts().to_string())

    # Print how many shots each player made
    print("\n--- Shots per Player ---")
    print(df["player"].value_counts().to_string())

    # Print detailed breakdown of shot tupes per player
    # This shows e.g. how many forehands Player 1 hit vs Player 2
    print("\n--- Shot Breakdown per Player ---")
    breakdown = df.groupby(["player", "shot_type"]).size().unstack(fill_value=0)
    print(breakdown.to_string())

    print("\nFiles saved to output/ folder")