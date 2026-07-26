//! The TUI in a real pseudo-terminal: launch it, type at it, and read the frames back.
//!
//! Everything else tests the app with the terminal taken away — `--plain` renders one
//! frame and exits, and the unit tests drive the mode handlers directly. Neither reaches
//! the part a user actually runs: raw mode, the alternate screen, mouse capture, the
//! event loop, and the restore on the way out. A crash or a hang there is invisible to
//! every other test in the suite and total for the user.
//!
//! Unix only (`openpty`). Each test spawns the real binary against the generated
//! fixture, waits for text to appear rather than sleeping a fixed time, and always
//! quits with `q` so the terminal restore path runs too.

#![cfg(unix)]

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const FIXTURE: &str = "tests/fixtures/tiny.safetensors";
/// Generous: a cold `cargo test` run may still be paging the binary in.
const WAIT: Duration = Duration::from_secs(20);

/// A running TUI plus the master side of its terminal.
struct Tui {
    child: Child,
    master: OwnedFd,
    seen: String,
    /// Where the output produced *since the last keypress* starts.
    ///
    /// `seen` accumulates every frame, and the TUI overwrites in place rather than
    /// scrolling, so a naive `contains` matches text the screen stopped showing several
    /// frames ago — which made one test "pass" while the search box was still open.
    /// Waits look only past this mark.
    mark: usize,
}

impl Tui {
    /// Launch the binary with `args` on an 80×24 pseudo-terminal.
    fn launch(args: &[&str]) -> Tui {
        let (master, slave) = openpty(24, 80);
        // The child gets the slave as all three streams, so crossterm sees a tty and
        // takes the interactive path.
        let child = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
            .args(args)
            .stdin(Stdio::from(slave.try_clone().expect("dup slave")))
            .stdout(Stdio::from(slave.try_clone().expect("dup slave")))
            .stderr(Stdio::from(slave))
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .spawn()
            .expect("spawn the TUI");
        Tui {
            child,
            master,
            seen: String::new(),
            mark: 0,
        }
    }

    /// Read until `needle` appears on screen, or fail.
    ///
    /// Compared with whitespace removed from both sides: the TUI positions the cursor
    /// per span rather than padding with spaces, so "Checkpoint Studio" arrives as
    /// `Checkpoint` + a cursor move + `Studio` and never contains the space itself.
    fn wait_for(&mut self, needle: &str) {
        let want = squash(needle);
        let deadline = Instant::now() + WAIT;
        while Instant::now() < deadline {
            if squash(&strip_ansi(&self.seen[self.mark..])).contains(&want) {
                return;
            }
            self.pump();
        }
        panic!(
            "timed out waiting for {needle:?}; screen was:\n{}",
            strip_ansi(&self.seen)
        );
    }

    /// Read whatever is available right now (short poll, so this can't hang).
    fn pump(&mut self) {
        let mut buf = [0u8; 8192];
        if !readable(&self.master, 200) {
            return;
        }
        // Safety: the fd is owned by `self` and stays open for the read.
        let mut f = unsafe { std::fs::File::from_raw_fd(libc::dup(self.master.as_raw_fd())) };
        if let Ok(n) = f.read(&mut buf) {
            self.seen.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
    }

    fn send(&mut self, keys: &str) {
        // Drain anything already pending, then mark: a wait after this call must be
        // satisfied by output the keypress caused, not by an older frame.
        self.pump();
        self.mark = self.seen.len();
        let mut f = unsafe { std::fs::File::from_raw_fd(libc::dup(self.master.as_raw_fd())) };
        f.write_all(keys.as_bytes()).expect("write keys");
        f.flush().ok();
    }

    /// Quit and assert a clean exit — which also exercises the terminal restore path.
    ///
    /// Getting out takes more than `q`: `q` only quits from the tree and the stats
    /// screen (on a sub-screen it steps back, deliberately, so a stray `q` can't drop
    /// you out of the app), a pop-up wants `Esc`, and in the rename screen `q` is text
    /// typed into a field. So unwind with Esc + Backspace before each `q`.
    fn quit(mut self) {
        // One key per pass, so each is processed (and any redraw drained) before the
        // next: a burst can otherwise arrive while the app is still in a modal state.
        for key in [
            "\u{1b}", "\u{7f}", "q", "\u{1b}", "\u{7f}", "q", "\u{7f}", "q", "q",
        ] {
            self.send(key);
            self.pump();
            self.pump();
            if self.child.try_wait().expect("wait").is_some() {
                break;
            }
        }
        let deadline = Instant::now() + WAIT;
        loop {
            match self.child.try_wait().expect("wait") {
                Some(status) => {
                    assert!(
                        status.success(),
                        "the TUI exited with {status:?}; screen was:\n{}",
                        strip_ansi(&self.seen)
                    );
                    return;
                }
                None if Instant::now() > deadline => {
                    let _ = self.child.kill();
                    panic!(
                        "the TUI did not exit on `q`; screen was:\n{}",
                        strip_ansi(&self.seen)
                    );
                }
                None => self.pump(),
            }
        }
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn openpty(rows: u16, cols: u16) -> (OwnedFd, OwnedFd) {
    let mut master = 0;
    let mut slave = 0;
    let size = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // Safety: both fds are out-params, and `size` is a valid winsize.
    let rc = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &size,
        )
    };
    assert_eq!(rc, 0, "openpty failed");
    // Safety: openpty returned two fresh, owned fds.
    unsafe { (OwnedFd::from_raw_fd(master), OwnedFd::from_raw_fd(slave)) }
}

fn readable(fd: &OwnedFd, millis: i32) -> bool {
    let mut p = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // Safety: one valid pollfd.
    unsafe { libc::poll(&mut p, 1, millis) > 0 }
}

/// Drop all whitespace, for comparing against a cursor-positioned screen.
fn squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn strip_ansi(s: &str) -> String {
    // Drop CSI/OSC escapes so assertions read against the visible text. The TUI
    // overwrites in place, so this is the union of everything drawn, not one frame.
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('[') => {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            Some(']') => {
                // OSC: ends at BEL or ST.
                while let Some(c) = chars.next() {
                    if c == '\u{7}' || (c == '\u{1b}' && chars.peek() == Some(&'\\')) {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn fixture() -> String {
    // `tests/cli.rs` generates this; make sure it exists for a standalone run of this
    // file (`cargo test --test pty`).
    if !std::path::Path::new(FIXTURE).exists() {
        let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
            .args(["--help"])
            .output();
        assert!(out.is_ok(), "the binary runs");
        panic!("{FIXTURE} is missing — run `cargo test --test cli` first (it generates it)");
    }
    FIXTURE.to_string()
}

#[test]
fn the_tui_starts_draws_the_tree_and_quits_cleanly() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.wait_for("tiny.safetensors");
    // The footer is drawn, so the whole frame (not just the header) made it out.
    tui.wait_for("quit");
    tui.quit();
}

#[test]
fn keys_navigate_between_screens_and_back() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");

    // Expand everything, then walk down past the groups and into a tensor's detail.
    // (Enter on a group folds it, so step a few rows first.)
    tui.send("e");
    for _ in 0..3 {
        tui.send("\u{1b}[B"); // ↓
    }
    tui.send("\r"); // Enter
    tui.wait_for("Tensor Details");

    // Backspace returns to the tree, `s` opens the stats screen.
    tui.send("\u{7f}");
    tui.wait_for("Checkpoint Studio");
    tui.send("s");
    tui.wait_for("Tensors");

    // Tab reaches the file browser from the tree.
    tui.send("\u{7f}");
    tui.send("\t");
    tui.wait_for("File browser");
    tui.send("\t");
    tui.wait_for("Checkpoint Studio");
    tui.quit();
}

#[test]
fn the_search_box_filters_as_it_is_typed() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.send("/");
    tui.send("norm");
    // The search box echoes the query and the tree narrows to matches.
    tui.wait_for("norm.weight");
    // Esc leaves the box; the tree's own footer comes back (while searching it shows
    // the search footer instead), which is how we know we're out of the box.
    tui.send("\u{1b}");
    tui.wait_for("expand/collapse");
    tui.quit();
}

#[test]
fn the_command_palette_opens_and_runs_a_command() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.send(":"); // palette
    tui.wait_for("Commands"); // the palette's title
    tui.send("stat"); // narrow to the stats command
    tui.send("\r");
    tui.wait_for("Tensors"); // the stats screen ran
    tui.quit();
}

#[test]
fn the_legend_and_the_y_command_overlay_and_dismiss() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.send("l");
    tui.wait_for("Legend");
    tui.send(" "); // any key closes it
    tui.send("y");
    tui.wait_for("checkpoint-studio"); // the reopen command is shown
    tui.send(" ");
    tui.quit();
}

#[test]
fn a_data_view_opens_from_the_detail_screen() {
    let mut tui = Tui::launch(&[
        &fixture(),
        "--tensor",
        "model.layers.0.mlp.down_proj.weight",
    ]);
    tui.wait_for("down_proj.weight");
    tui.send("v"); // the numeric grid
    tui.wait_for("Values");
    tui.send("b"); // cycle the base
    tui.send("z"); // cycle the zebra striping
    tui.send("\u{1b}[C"); // pan right
    tui.send("m"); // switch to the heatmap
    tui.wait_for("Heatmap");
    tui.quit();
}

#[test]
fn a_resize_is_handled_rather_than_crashing() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    // Shrink to something smaller than the chrome, then grow again. Ratatui panics on
    // an out-of-bounds write, so this is the regression test for the clamped rects.
    for (rows, cols) in [(6u16, 30u16), (2, 12), (40, 200), (24, 80)] {
        let size = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // Safety: a valid winsize on the master fd.
        unsafe {
            libc::ioctl(tui.master.as_raw_fd(), libc::TIOCSWINSZ, &size);
            libc::kill(tui.child.id() as i32, libc::SIGWINCH);
        }
        tui.send("e"); // force a redraw at the new size
        tui.pump();
        assert!(
            tui.child.try_wait().expect("wait").is_none(),
            "the TUI died at {cols}x{rows}; screen was:\n{}",
            strip_ansi(&tui.seen)
        );
    }
    tui.quit();
}

/// Send an SGR mouse press+release at 1-based `(col, row)`, the way a terminal does.
fn click(tui: &mut Tui, col: u16, row: u16) {
    tui.send(&format!("\u{1b}[<0;{col};{row}M"));
    tui.send(&format!("\u{1b}[<0;{col};{row}m"));
}

#[test]
fn clicking_a_row_and_the_wheel_move_the_tree() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.send("e");
    // A click on a tree row selects it; the wheel scrolls. Mouse capture is on, so
    // these arrive as SGR sequences — a path no unit test can reach.
    // A click may select, fold or open depending on the row, so assert what matters:
    // the mouse path runs and the app stays alive.
    click(&mut tui, 6, 5);
    tui.send("\u{1b}[<65;10;10M"); // wheel down
    tui.send("\u{1b}[<64;10;10M"); // wheel up
    tui.pump();
    assert!(
        tui.child.try_wait().expect("wait").is_none(),
        "the TUI died handling the mouse; screen was:\n{}",
        strip_ansi(&tui.seen)
    );
    tui.quit();
}

#[test]
fn the_file_browser_opens_a_layout_map_and_comes_back() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.send("\t");
    tui.wait_for("File browser");
    // Walk to the fixture row and open it: a `.safetensors` file opens its byte layout.
    for _ in 0..3 {
        tui.send("\u{1b}[B");
    }
    tui.send("\r");
    tui.pump();
    tui.send("\u{7f}"); // back
    tui.pump();
    tui.quit();
}

#[test]
fn the_health_report_opens_over_the_tree() {
    let mut tui = Tui::launch(&[&fixture()]);
    tui.wait_for("Checkpoint Studio");
    tui.send("h");
    // The check report runs and lists its checks.
    tui.wait_for("Shape / dtype sanity");
    tui.send("\u{7f}");
    tui.quit();
}

#[test]
fn the_dtype_and_reshape_prompts_open_and_cancel() {
    let mut tui = Tui::launch(&[
        &fixture(),
        "--tensor",
        "model.layers.0.mlp.down_proj.weight",
    ]);
    tui.wait_for("Tensor Details");
    tui.send("d"); // dtype menu
    tui.pump();
    tui.send("\u{1b}"); // cancel
    tui.send("r"); // reshape prompt
    tui.wait_for("Reshape");
    tui.send("\u{1b}");
    tui.pump();
    tui.quit();
}

#[test]
fn the_stats_screen_scrolls_and_folds_its_shard_breakdown() {
    let mut tui = Tui::launch(&[&fixture(), "--stats"]);
    tui.wait_for("Tensors");
    tui.send("f"); // fold/expand the per-shard breakdown
    tui.pump();
    tui.send("\u{1b}[B"); // scroll
    tui.send("\u{1b}[6~"); // PgDn
    tui.send("\u{1b}[5~"); // PgUp
    tui.pump();
    tui.quit();
}

#[test]
fn the_rename_screen_opens_and_leaves_without_writing() {
    // `R` is the only screen that can modify a checkpoint; opening and leaving it must
    // not touch the file.
    let path = fixture();
    let before = std::fs::metadata(&path).expect("the fixture exists");
    let mut tui = Tui::launch(&[&path]);
    tui.wait_for("Checkpoint Studio");
    tui.send("R");
    tui.wait_for("Rename");
    tui.send("\u{7f}"); // back out
    tui.pump();
    tui.quit();
    let after = std::fs::metadata(&path).expect("still there");
    assert_eq!(before.len(), after.len(), "opening rename must not write");
}
