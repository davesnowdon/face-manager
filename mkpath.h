/*
 * Define mkpath function from https://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
 */
extern "C" {
int mkpath(const char *path, mode_t mode);
}
