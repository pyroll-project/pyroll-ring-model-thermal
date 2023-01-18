$pdf_mode = 4;
ensure_path("TEXINPUTS", ".", ".build/");
$do_cd=1;
$silent_mode=1;
set_tex_cmds( "--shell-escape" );