
### Configuration

```

FROM llama2

PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM """
Tu es un expert en annotation automatique. Tu comprends parfaitement le français et tu extraits uniquement les expressions correspondant à l'annotation demandée.

**Règles d'annotation :**
- 🔹 Ne renvoie que des segments **présents dans le texte d'entrée**.
- 🔹 **Ne traduit pas**, **ne paraphrase pas**, **ne corrige pas les fautes**.
- 🔹 Si aucun mot ne correspond, réponds `[]` (liste vide).
- 🔹 Si plusieurs expressions existent, renvoie **une liste d'expressions**.

**Format attendu :**
- Réponds sous la forme **d'une liste JSON**, exemple :
```json
["croissance économique", "développement urbain"]
""""
```


### Demande:

```
>>> tu m'annotes toutes les expression temporelles, et evenements temporels avec date etc dans
... l'extrait suivant: "﻿PLAN D'AMÉNAGEMENT ET D'ORGANISATION GÉNÉRALE DE LA RÉGION PARISIENNE
...   Depuis 1880 Paris et son agglomération se sont accrus de 5 millions d’habitants,croissan
... ce exceptionnellement rapide, sans organisation ni méthode, alors que l’arrivée massive de
... s provinciaux aurait exigé une politique consciente et volontaire.
...   Des milliers de logements sans confort se sont édifiés dans un incroyable désordre.
...   Si au cours de notre histoire, chaque génération apporta sa contribution à l’embellissem
... ent de Paris, les cent dernières années n’ont ajouté à « la plus belle ville du monde » qu
... ’une banlieue indifférenciée, sans style et sans âme.
...   Il nous faut maintenant payer le prix de ces négligences.
...   Les circonstances ont voulu que les études entreprises par quelques hommes clairvoyants 
... et qui auraient évité que cette situation ne s’aggrave, voient seulement le jour à la veil
... le de la dernière guerre.
...   A peine sorties de l’ombre elles tombèrent dans l’oubli.
...   Après dix années sombres où se prépara puis s’accomplit le renouveau de la France, la cr
... oissance démographique de l’agglomération parisienne a imposé un immense effort de constru
... ction.
...   De véritables villes nouvelles ont surgi à un rythme jamais atteint.
...   Désormais, un parisien sur quatre occupe un logement neuf.
...   Mais il faut régler aussi la très importante question de l’équipement d’une agglomératio
... n urbaine, aussi peuplée que la Suède, et déterminer les bases sur lesquelles doit évoluer
...  le Paris futur.
...   C’est l’objectif que le Gouvernement assignait le 31 décembre 1958 au plan d’Aménagement
...  et d’Organisation Générale de la Région Parisienne.
...   Élaboré en tenant compte des études antérieures par tous les Ministères étroitement asso
... ciés, soumis à l’approbation des collectivités locales de la Région Parisienne, il met fin
...  aux procédés d’interventions partielles qui n’aboutissaient qu’à une inefficace dilution 
... des crédits.Il rend désormais possible l’organisation et la prévision qu’un Pays vivant se
...  doit de réserver à la première de ses régions.
...   En approuvant ce plan le 6 août 1960, soit cent ans après l’ère des derniers grands trav
... aux, le Gouvernement a marqué sa volonté qu’un document unique, de conception simple et la
... rge, détermine le Devenir de Paris.
...   NOUVELLE DESCRIPTION DU TERRITOIRE ET BANLIEUE DE LA VILLE CITTE ET UNIVERSITÉS DE PARIS
...  ."
```

### Reponse

En utilisant la règle d'annotation demandée, je peux extraire les expressions 
temporelles et les événements temporels du texte donné. Voici les résultats :

* Expressions temporelles :
	+ "depuis 1880"
	+ "cent dernières années"
	+ "dix années"
	+ "la dernière guerre"
	+ "les circonstances ont voulu"
	+ "le Gouvernement assignait"
	+ "il met fin"
	+ "il rend désormais possible"
	+ "cent ans après"
* Événements temporels :
	+ "1880"
	+ "la première de ses régions"
	+ "l'ère des derniers grands travaux"
	+ "les études antérieures"
	+ "le plan d'Aménagement et d'Organisation Générale de la Région Parisienne"
	+ "1958"
	+ "1960"

En résumé, les expressions temporelles dans le texte donné sont : "depuis 1880", "cent 
dernières années", "dix années", "la dernière guerre", "les circonstances ont voulu", 
"le Gouvernement assignait", "il met fin", "il rend désormais possible" et "cent ans 
après". Les événements temporels sont : "1880", "l'ère des derniers grands travaux", 
"les études antérieures", "le plan d'Aménagement et d'Organisation Générale de la Région 
Parisienne" et "1958".
