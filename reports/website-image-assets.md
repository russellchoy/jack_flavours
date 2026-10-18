# Gelato animation assets

The website uses an original transparent photographic sprite atlas generated with the built-in ImageGen tool on 18 October 2026. Asset: `web/assets/gelato-atlas.png`. The daily dashboard copies `web/assets` into its published site so scheduled refreshes retain the animation.

User-supplied reference: Scott Grummett’s Jack’s Gelato group photograph (white paper cups, textured gelato and chocolate-chip flecks), plus photographs of a served cup, a chocolate-dipped cone and a hand building a tall scoop stack. These guided art direction; the reference photographs are not republished.

Generation prompt: Create a transparent square 2×2 photographic sprite atlas with four separate equal cells: empty white paper gelato cup; raspberry gelato scoop; pale pistachio scoop; human right hand holding an empty stainless-steel ice-cream scoop. Consistent elevated three-quarter view, soft upper-left lighting, realistic handmade gelato texture, no logo, text, backdrop or cell overlap.

Final edit prompt: Change only the bottom-left scoop to ivory-white stracciatella with irregular dark chocolate shards, preserving creamy folds, natural scoop ridges, position, scale and silhouette. Preserve the other cells and transparent background. Keep the metal scoop within the bottom-right quadrant.

The frontend isolates the cells in CSS, clips a small neighbouring-cell edge, and synchronizes their transforms to ordinary scroll position. Reduced-motion preferences show a completed cup without animation. Images are illustrative, not photographs of the predicted flavours or a claim of availability.
