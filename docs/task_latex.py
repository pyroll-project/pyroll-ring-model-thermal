import pytask


@pytask.mark.latex(
    script="docs.tex",
    document="docs.pdf",
)
def task_latex():
    pass
