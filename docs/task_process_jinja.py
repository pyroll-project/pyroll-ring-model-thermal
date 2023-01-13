import jinja2
import pytask
from pathlib import Path
import toml

from docs.config import BUILD_DIR, ROOT_DIR

JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT_DIR, encoding="utf-8"))


def jinja_task(tex_file: Path | str):
    tex_file = (Path(ROOT_DIR / tex_file) if isinstance(tex_file, str) else tex_file).resolve()

    def dec(func):
        @pytask.mark.task(id=str(tex_file.relative_to(ROOT_DIR)))
        @pytask.mark.depends_on({
            "template": tex_file,
        })
        @pytask.mark.produces(BUILD_DIR / tex_file.relative_to(ROOT_DIR))
        def task_process_jinja(depends_on: dict[str, Path], produces: Path):
            template = JINJA_ENV.get_template(str(depends_on["template"].relative_to(ROOT_DIR)))

            result = template.render(func(depends_on=depends_on))

            produces.write_text(result)

        return task_process_jinja

    return dec


@pytask.mark.depends_on({
    "data": "symbols.toml"
})
@jinja_task("symbol_index.tex")
def symbol_index(depends_on: dict[str, Path]):
    symbols = toml.loads(depends_on["data"].read_text())["symbol"]

    def get_sort_key(e):
        return (
            (e.get("sort", None) or e["code"])
            .removeprefix("\\Delta")
            .removeprefix("\\dot")
            .removeprefix("\\hat")
            .strip("\\ {").lower()
            .removeprefix("var")
        )

    symbols = sorted(symbols, key=get_sort_key)
    return dict(
        symbols=symbols
    )
