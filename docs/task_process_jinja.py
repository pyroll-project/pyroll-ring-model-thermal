import jinja2
import pytask
from pathlib import Path
import toml

THIS_DIR = Path(__file__).parent

JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(THIS_DIR, encoding="utf-8"))


def jinja_task(tex_file: Path | str):
    tex_file = (Path(THIS_DIR / tex_file) if isinstance(tex_file, str) else tex_file).resolve()

    def dec(func):
        @pytask.mark.task(id=str(tex_file.relative_to(THIS_DIR)))
        @pytask.mark.depends_on({
            "template": tex_file,
        })
        @pytask.mark.produces(THIS_DIR / ".build" / tex_file.relative_to(THIS_DIR))
        def task_process_jinja(depends_on: dict[str, Path], produces: Path):
            template = JINJA_ENV.get_template(str(depends_on["template"].relative_to(THIS_DIR)))

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
    return dict(
        symbols=symbols
    )
