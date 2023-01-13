import pytask

from docs.config import BUILD_DIR


@pytask.mark.latex(
    script="docs.tex",
    document="docs.pdf",
)
@pytask.mark.depends_on(BUILD_DIR.rglob("*"))
def task_latex():
    pass
