import os
import xml.etree.ElementTree as ET
import pandas as pd


def save_to_csv(data, output_file):
    """
    Enregistre les phrases extraites du fichier XML dans un fichier CSV.

    Args:
        data (tuples): Liste contenant (phrase, 0 (s'il n'y a pas de dynamique) ou 1 (s'il y en a une)).
        output_file (str): Chemin du fichier de sortie.
    """
    if data:
        df = pd.DataFrame(data, columns=["sentence", "dynamic"])
        df.to_csv(output_file, sep="|")
        print(f"Le fichier .csv: {output_file} a été enregistré.")


def process_dynamic_xmlfile(dossier_input, dossier_output):
    """
    Parcourt tous les fichiers XML d'un dossier, extrait les phrases.
    Création d'un tuple (phrase, 0 (s'il n'y a pas de dynamique) ou 1 (s'il y en a une)).
    Enregistre en CSV.

    Args:
        dossier_input (str): Dossier contenant les fichiers XML.
        dossier_output (str): Dossier où les fichiers CSV sont enregistrés.
    """
    for fichier_xml in os.listdir(dossier_input):
        if fichier_xml.endswith(".xml"):
            xml_path = os.path.join(dossier_input, fichier_xml)
            tree = ET.parse(xml_path)
            root = tree.getroot()

        data = []

        for phrase in root.findall(".//phrase"):
            no_tags_text = "".join(phrase.itertext()).strip()

            if phrase.find(".//dyn") is not None:
                data.append((no_tags_text, 1))
            else:
                mots = no_tags_text.split()
                if len(mots) > 10:  # on définit qu'une phrase c'est 10 mots
                    data.append((no_tags_text, 0))

        output_file = os.path.join(dossier_output, fichier_xml.replace(".xml", ".csv"))
        save_to_csv(data, output_file)


# Lancement
dossier_input = "../corpus/corpus_annote"
dossier_output = "../corpus/corpus_phrases"
process_dynamic_xmlfile(dossier_input, dossier_output)
