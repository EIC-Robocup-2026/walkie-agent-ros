# 2. Add the Documentation Site

**Goal:** build this website from Markdown files in the `docs/` folder, and
publish it to GitHub Pages automatically.

We use [MkDocs](https://www.mkdocs.org/) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. You write
normal Markdown, and MkDocs turns it into a website.

## Step 1: Install MkDocs in its own environment

```bash
pixi add --feature docs "mkdocs-material>=9.7,<10" "mkdocs>=1.6,<2"
pixi workspace environment add docs --feature docs --no-default-feature
pixi install -e docs
```

What these do:

- A **feature** is a named group of packages. `--feature docs` puts MkDocs
  in a group called `docs`.
- An **environment** is a set of features that gets installed together. We
  made an environment called `docs` that contains only the `docs` feature.
- `--no-default-feature` keeps the ROS packages out of the `docs`
  environment, and keeps MkDocs out of the ROS environment. They never
  conflict with each other.
- `pixi install -e docs` installs the `docs` environment.

!!! info "Why `mkdocs<2`?"
MkDocs 2.0 does not work with the Material theme. The version limit makes
sure Pixi never installs it by accident.

## Step 2: Add tasks

Tasks are shortcuts for long commands. We added these to `pixi.toml`:

```toml title="pixi.toml"
[feature.docs.tasks]
docs = "mkdocs serve"
docs-build = "mkdocs build --strict"
```

Now you can run:

| Command               | What it does                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| `pixi run docs`       | Preview the site at <http://127.0.0.1:8000>. Reloads when you save.                                |
| `pixi run docs-build` | Build the final site into `site/`. `--strict` makes it fail on any warning, such as a broken link. |

These tasks only exist in the `docs` environment, so Pixi picks that
environment automatically.

## Step 3: Configure the site

`mkdocs.yml` in the project root controls the site. The important parts:

```yaml title="mkdocs.yml"
site_name: Walkie Agent ROS # title shown at the top

theme:
  name: material # use the Material theme

nav: # the menu on the left
  - Home: index.md
  - Getting Started: getting-started.md
```

The rest of the file turns on extra features, like the dark mode toggle,
copy buttons on code blocks, and the colored note boxes you see on this page.

We also added `site/` to `.gitignore`, because it is build output.

## Step 4: Publish with GitHub Actions

The file `.github/workflows/docs.yml` runs on GitHub every time someone
pushes changes to `docs/`, `mkdocs.yml`, or `pixi.toml` on `main`. It:

1. Installs Pixi and the `docs` environment.
2. Runs `pixi run docs-build`.
3. Uploads `site/` to GitHub Pages.

!!! warning "One-time setting on GitHub"
In the repository, go to **Settings → Pages** and set **Source** to
**GitHub Actions**. Without this, the publish step fails.

## Writing a new page

1. Create a Markdown file in `docs/`, for example `docs/my-topic.md`.
2. Add it to `nav` in `mkdocs.yml`:

   ```yaml
   nav:
     - Home: index.md
     - My Topic: my-topic.md
   ```

3. Run `pixi run docs` and check it in your browser.
4. Run `pixi run docs-build` before you commit. If it fails, fix the warning
   it prints.

Some useful things you can write in a page:

=== "Note box"

    ```markdown
    !!! note "Title"
        Text inside the box. Indent it with 4 spaces.
    ```

    Other box types: `tip`, `info`, `warning`, `danger`.

=== "Code block with title"

    ````markdown
    ```bash title="terminal"
    pixi run docs
    ```
    ````

=== "Tabs"

    ```markdown
    === "Tab one"

        Content of tab one.

    === "Tab two"

        Content of tab two.
    ```

See the [Material reference](https://squidfunk.github.io/mkdocs-material/reference/)
for everything else.
