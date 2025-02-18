### Configuration 

````
FROM llama3.1

PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM """
Tu es un expert en urbanisme réquisitionné pour un travail d'annotation de lieux. Tu comprends parfaitement le français. 
Extrait tous les noms de lieux sans en exclure aucun en fonction du contexte du texte d'entrée. 
Ils peuvent être des noms de lieux autonomes, mais aussi faire partie intégrante de la description de lieux spécifiques dans le texte.

**Règles**
- Extraire **tous** les noms de lieux (noms propres et noms communs désignant des lieux) présents dans le texte.
- Si aucun nom de lieu n'existe, renvoyer `[]` (liste vide).
- La sortie doit être au **format de liste JSON**.

**Format de sortie:**
-La réponse doit être une une seule liste JSON contenant toutes les instances de lieux repérées dans le texte donné. 
example:
```json
["la rue de ma maison", "Paris"]
"""
````

### Demande
```
>>> Notre amitié fut immédiate, et je lui proposai de me rejoindre à l’AUA, au sein duquel nous formerons un peu plus tard, avec Borja Huidobro, le groupe « CCH, paysagistes urbains ».
...   C’est le projet du quartier de l’Arlequin à la Villeneuve de Grenoble qui nous occupe le plus.
...   Georges Loiseau et Jean Tribel, architectes à l’AUA, conduisaient cette opération.
...   Nous devions en étudier la « rue » et le parc au centre du quartier.
...   Ce projet revendiquait une forte densité, une grande continuité formelle du bâti.
...   Il voulait créer de véritables espaces publics : la rue, le parc, une distribution qui favorisait la mixité, l’intégration des équipements à partir de la rue.
...   Il proposait la mise en place, dans les écoles, de nouvelles pédagogies.
...   Il espérait réduire la ségrégation sociale.
...   Cette utopie a, pour partie, fonctionné quelques années, puis s’est progressivement défaite.
...   On peut critiquer, aujourd’hui, ce « grand ensemble » l’échec relatif de la rue désactivée par l’arrivée du grand centre commercial, l’enfermement sur elle-même de cette forme urbaine singulière, coupée des quartiers voisins.
...   Coupure qui, aujourd’hui, renforce les effets de ghetto que l’homogénéité sociale a progressivement initiée.
...   Mais je n’ai plus jamais connu une telle intensité, une telle complémentarité de réflexions.
...   Quant au projet du parc de vingt hectares de la Villeneuve, il témoigne, le premier, de mon attachement à la campagne.
...   J’écrivais alors : « Le choc entre géométrie et géographie préside bien à la conception du parc de la Villeneuve mais dans un ordre inversé des efforts qui ont façonné la campagne.
...   La géométrie n’est pas, ici, la figure qui se déploie sur un fond, elle est le fond lui-même, le substrat, le site d’origine.
...   Sur ces terrains, rigoureusement plans, enclos par l’architecture, dans l’espace de cette sorte de clairière se tissaient déjà les lignes qui vont et viennent de l’ombre épaisse des bâtiments.
...   L’espace est chargé des traits de la ville et je n’ai eu qu’à les graver sur le sol.
...   Grenoble, parc de la Villeneuve « Pour bien exprimer que l’ordre qui préside à ces lieux est celui de la ville, les chemins et les arbres traversent l’espace en gardant leur alignement.
...   Ils franchissent la forme des buttes soit pour la blesser, soit pour s’y soumettre, car c’est la campagne qu’ils doivent, à leur tour, évoquer ».
...   Maîtrise d’ouvrage: agence d’urbanisme de la région Grenobloise Programme: parc urbain du quartier de l’Arlequin
```

### Réponse

```
Voici la liste des noms de lieux extraits du texte :

json
[
  "AUA",
  "quartier de l'Arlequin",
  "Villeneuve de Grenoble",
  "Grenoble",
  "rue",
  "parc au centre du quartier",
  "centre commercial",
  "campagne",
  "parc de la Villeneuve"
]

J'ai inclus tous les noms de lieux mentionnés dans le texte, qu'ils soient des noms propres (comme Grenoble) ou des noms communs désignant des types de lieux (comme rue ou parc).
```