# 768-D Embedding Path Visualizer

An interactive geometric encoding of a deterministic example 768-dimensional
vector. Each scalar becomes one segment in a 3D path:

- `abs(value)` controls segment length.
- A positive value makes a right-hand turn.
- A negative value makes a left-hand turn.
- The selected consecutive angle controls the exact angle between adjacent
  segments.
- An index-driven roll rotates the local turning plane, giving the path genuine
  3D structure without changing the meaning of the sign.

The visualization rotates around the z-axis and supports orbiting, zooming,
click-to-deform interaction, point-count resampling, angle changes, shape tools,
colors, line width, point size, and rotation-speed controls.

## Live static page

The ready-to-host site is [`docs/index.html`](docs/index.html). It has no server
or external JavaScript dependencies.

To preview it locally, either open that file directly or run:

```bash
python3 -m http.server 8000 --directory docs
```

Then visit <http://localhost:8000>.

## Run the Python generator locally

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/FL2744/EMBED.git
cd EMBED
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python EMBED.py
```

The script creates `embedding_visualization.html` and opens it in the default
browser. The generated file is intentionally ignored by Git because the
publishable copy is maintained at `docs/index.html`.

Only NumPy is required for normal terminal or VS Code execution. For a local
Jupyter or VS Code notebook environment, install the additional notebook
dependencies:

```bash
python -m pip install -r requirements-notebook.txt
```

In VS Code, select `.venv/bin/python` with **Python: Select Interpreter** before
using Run or Debug.

## Google Colab

Upload `EMBED.py`, then run:

```python
%run /content/EMBED.py
```

The visualization renders inline in Colab, JupyterLab, and VS Code notebooks.

## Publish with GitHub Pages

1. Push this folder to a GitHub repository.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the default branch and the `/docs` folder, then save.

The site will normally be available at:

```text
https://YOUR-USERNAME.github.io/REPOSITORY-NAME/
```

## Embed in Canvas LMS

Canvas course pages generally do not execute pasted JavaScript. Publish the
`docs` page externally and embed its HTTPS address instead:

```html
<iframe
  src="https://YOUR-USERNAME.github.io/REPOSITORY-NAME/"
  width="100%"
  height="760"
  style="border:0"
  loading="lazy"
  allowfullscreen>
</iframe>
```

If the institution removes the iframe or blocks the domain, add the page as an
External URL module item or ask the Canvas administrator to allow the domain.

## Reconstruction limits

The original vector can be reconstructed from the undeformed 3D coordinates
when the roll rule, angle, scale, and all 769 vertices are retained. Reducing
the point count, smoothing, twisting, or deforming the path discards or changes
encoded information. A screenshot is not sufficient for exact reconstruction.

## Repository layout

```text
.
├── EMBED.py               # Generator and notebook/local application
├── docs/
│   └── index.html         # Standalone GitHub Pages build
├── requirements.txt
├── requirements-notebook.txt
├── .gitignore
└── README.md
```

## License

No license has been selected. Add a license before inviting reuse or
contributions.
