##__IMPORTS__##
from lxml import etree
import glob

CHEMIN_ALL_FILES = glob.glob("corpus/corpus_annote/*.xml")


def delete_tags():

    for files in CHEMIN_ALL_FILES:
        with open(files, "r") as f:
            text = f.read()

        # Parse the XML
        root = etree.fromstring(text)
        virgin_text = "".join(root.itertext())

        nouveau = files.split("/")[-1].split(".")[0]

        new_path = f"corpus/corpus_vierge/{nouveau}.txt"

        with open(new_path, "w") as output_file:
            output_file.write(virgin_text)


delete_tags()
