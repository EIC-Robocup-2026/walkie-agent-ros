/*
 * Loads rmw_zenoh_cpp 0.2.11 into the processes that run under Zenoh.
 *
 * scripts/activate.sh preloads this library into every process of a pixi
 * session. When a process starts, and RMW_IMPLEMENTATION is rmw_zenoh_cpp,
 * the library opens the 0.2.11 build in rmw_zenoh_ws/install. ROS later opens
 * the middleware by its file name, librmw_zenoh_cpp.so, and reuses the library
 * that is already loaded under that name instead of RoboStack's 0.2.10
 * (ADR-0032, ADR-0033).
 *
 * The check runs when the process starts, after pixi has applied
 * [activation.env] and the task's env, so both can choose the middleware.
 * 0.2.11 is opened with RTLD_LOCAL: its functions do not replace those of
 * librmw_implementation.so, so a process that switches to another middleware
 * before it imports rclpy, as tests/conftest.py does, still gets that
 * middleware.
 *
 * The path comes from PIXI_PROJECT_ROOT. LD_PRELOAD is passed on to every
 * child process, and a `pixi run` in another project sets its own root, so
 * that project's processes do not load this build. They have no such file,
 * and the library prints nothing for them.
 *
 * If the file exists but cannot be opened, the library prints the reason to
 * stderr. ROS then opens RoboStack's 0.2.10, as it would without this
 * library.
 *
 * Built by `pixi run build-rmw-zenoh`.
 */
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

__attribute__((constructor)) static void load_rmw_zenoh(void)
{
    const char *rmw = getenv("RMW_IMPLEMENTATION");
    const char *root = getenv("PIXI_PROJECT_ROOT");
    if (rmw == NULL || strcmp(rmw, "rmw_zenoh_cpp") != 0 || root == NULL) {
        return;
    }

    char path[PATH_MAX];
    int length = snprintf(path, sizeof path,
                          "%s/rmw_zenoh_ws/install/rmw_zenoh_cpp/lib/librmw_zenoh_cpp.so", root);
    if (length < 0 || (size_t)length >= sizeof path) {
        return;
    }
    if (dlopen(path, RTLD_LAZY | RTLD_LOCAL) == NULL && access(path, F_OK) == 0) {
        fprintf(stderr,
                "rmw_zenoh_loader: cannot load rmw_zenoh_cpp 0.2.11: %s. "
                "Zenoh nodes use RoboStack's 0.2.10, which can hang (ADR-0032).\n",
                dlerror());
    }
}
