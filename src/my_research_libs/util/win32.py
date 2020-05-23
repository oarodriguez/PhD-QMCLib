import ctypes
import platform


def enable_virtual_terminal_processing():
    """Set mode ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004) in Windows
     consoles.

     According to Microsoft Docs:
        When writing with WriteFile or WriteConsole, characters are
        parsed for VT100 and similar control character sequences that
        control cursor movement, color/font mode, and other operations
        that can also be performed via the existing Console APIs.

     See https://docs.microsoft.com/en-us/windows/console/setconsolemode
     for more details.

     NOTE 1: This implementation is based on the following resource:
        https://gitlab.com/dslackw/colored/-/merge_requests/12

     NOTE 2: IMO, the approach used by colored module to enable ANSI escape
        codes on Windows is somewhat invasive. Setting mode to
        ENABLE_VIRTUAL_TERMINAL_PROCESSING should be done explicitly
        by the user, not automatically by any library.
     """
    # Note: Cygwin should return something like "CYGWIN_NT..."
    if platform.system().lower() == 'windows':
        invalid_handle_value = ctypes.c_void_p(-1).value
        std_output_handle = ctypes.c_int(-11)
        std_handle = ctypes.windll.kernel32.GetStdHandle(std_output_handle)
        if std_handle == invalid_handle_value:
            return
        mode = ctypes.c_int(0)
        success = ctypes.windll.kernel32.GetConsoleMode(
                ctypes.c_int(std_handle),
                ctypes.byref(mode))
        if not success:
            return
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING: 0x0004
        target_mode = ctypes.c_int(mode.value | 0x0004)
        success = ctypes.windll.kernel32.SetConsoleMode(
                ctypes.c_int(std_handle), target_mode)
        if not success:
            # Check kernel32.GetLastError for more information.
            return
