import pandas as pd
import glob
from bs4 import BeautifulSoup


PATH_TSV = "annotations/annotations_auto"
PATH_TXT = f"{PATH_TSV}/raw"


def fill_holes(df, text_file):
    """
    Fills missing annotation columns in the dataframe using the structured text file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to update.
    - text_file (str): Path to the text file containing structured annotations.

    Returns:
    - pd.DataFrame: The updated DataFrame with missing annotations filled.
    """
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()
    soup = BeautifulSoup(text, "html.parser")

    # Extract annotations
    lieux_obj = [tag.text.strip() for tag in soup.find_all("lieu", {"type": "obj"})]
    lieux_loc = [tag.text.strip() for tag in soup.find_all("lieu", {"type": "loc"})]
    acts = [tag.text.strip() for tag in soup.find_all("act")]

    # Create DataFrame
    data = {
        "balise_lieu_obj": lieux_obj,
        "balise_lieu_loc": lieux_loc,
        "balise_act": acts,
    }

    # Convert to DataFrame and handle missing values
    new_df = pd.DataFrame.from_dict(data, orient="index").transpose()

    print(new_df)

    # Append new annotations to the existing DataFrame
    df = pd.concat([df, new_df], ignore_index=True)

    return df


def main():
    all_tsv = glob.glob(f"{PATH_TSV}/*.tsv")
    all_txt = glob.glob(f"{PATH_TXT}/*.txt")

    for file in all_txt:
        basename = file.split("/")[-1].split(".")[0]
        corresponding_tsv = f"{PATH_TSV}/{basename}.tsv"

        if corresponding_tsv in all_tsv:
            try:
                df = pd.read_csv(corresponding_tsv)
            except pd.errors.ParserError:
                continue
            filled_df = fill_holes(df, file)
            print(df, filled_df)

            # Save the updated DataFrame
            filled_df.to_csv(f"{PATH_TSV}/{basename}.tsv", index=False, sep="|")
            print(f"✅ Updated: {corresponding_tsv}")


if __name__ == "__main__":
    main()
