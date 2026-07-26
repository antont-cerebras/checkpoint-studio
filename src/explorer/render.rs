//! Drawing the detail, statistics and data screens — the frame each `Mode` asks for.
//!
//! Split out of `explorer/mod.rs` as a fourth `impl Explorer` block. Each screen has a
//! pair: the interactive frame (into a Ratatui buffer) and the `--plain` one-shot render
//! that the insta snapshots capture. Keeping the pairs together is deliberate: the
//! snapshots are the only automated check that the interactive drawing is right, so the
//! two must not drift apart.

#[allow(clippy::wildcard_imports)] // a submodule of the module it was split from
use super::*;

impl Explorer {
    /// Build the detail-screen draw config and render it into `frame` — the
    /// Ratatui counterpart of [`Self::render_tree_frame`], shared by the live loop
    /// and the headless render.
    #[allow(clippy::too_many_arguments)]
    // mirrors the screen renderer's params
    pub(super) fn render_detail_frame(
        &self,
        frame: &mut ratatui::Frame,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        hist_scanning: Option<crate::ui::ScanProgress>,
        overlay: Option<&Overlay>,
    ) {
        let (chips, links) = UI::render_detail(
            frame,
            tensor,
            shape,
            view,
            overridable,
            unindexed,
            stats,
            histogram,
            hist_scanning,
            self.schema_for(&tensor.name),
            overlay,
        );
        *self.clickable.borrow_mut() = chips;
        *self.links.borrow_mut() = links; // `File:` path → layout map
        let badges = self.screen_badges(HelpCtx::Detail);
        UI::render_badge_bar(frame, &badges, self.hovered_badge.get());
        self.render_shortcut_hover(frame);
    }

    /// Render the full-screen checkpoint-stats view (chrome + scrollable body +
    /// footer via [`UI::render_stats_frame`]), record its clickable regions, and
    /// draw the access badge on the reserved bottom row — the stats analogue of
    /// [`Self::render_detail_frame`]. Returns the maximum valid scroll offset so
    /// the mode can clamp its own.
    pub(super) fn render_stats_screen(
        &self,
        frame: &mut ratatui::Frame,
        stats: &crate::stats::CheckpointStats,
        scroll: usize,
        shards_expanded: bool,
    ) -> usize {
        let (max, regions, vscroll) = UI::render_stats_frame(frame, stats, scroll, shards_expanded);
        *self.clickable.borrow_mut() = regions;
        *self.vscrollbar.borrow_mut() = vscroll;
        let badges = self.screen_badges(HelpCtx::Stats);
        UI::render_badge_bar(frame, &badges, self.hovered_badge.get());
        max
    }

    /// Render a tensor's detail screen to plain text via an in-memory Ratatui
    /// backend — the headless (`--plain`) detail and the `c` screen-copy share
    /// this. Mirrors [`Self::tree_plain`].
    #[allow(clippy::too_many_arguments)] // mirrors the screen renderer's params
    pub(super) fn detail_plain(
        &self,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        overlay: Option<&Overlay>,
    ) -> Result<String> {
        crate::tui::headless_render(120, 40, |f| {
            self.render_detail_frame(
                f,
                tensor,
                shape,
                view,
                overridable,
                unindexed,
                stats,
                histogram,
                None,
                overlay,
            )
        })
    }

    /// Sample and render a tensor's data view (heatmap / numeric grid) into
    /// `frame`, compositing a pop-up `overlay` (legend / copied command) last —
    /// the Ratatui counterpart of [`Self::render_detail_frame`], shared by the
    /// live loop and the headless render. Returns `(slices, overridable,
    /// clamped_slice)` (or the error message [`Self::draw_data_view`] would
    /// have), so the loop can clamp the slice and gate slice/dtype hints.
    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    /// The data view's current sampling mode, from the session-remembered layout
    /// prefs (overview / edges split / window offset).
    pub(super) fn data_sample_mode(&self) -> SampleMode {
        match self.data_view.data_view_layout.get() {
            DataLayout::Edges => SampleMode::Edges {
                row_tail: self.data_view.data_view_row_tail.get(),
                col_tail: self.data_view.data_view_col_tail.get(),
            },
            DataLayout::Overview => SampleMode::Grid,
            DataLayout::OverviewMax => SampleMode::GridMax,
            DataLayout::Window => SampleMode::Window {
                row_off: self.data_view.data_view_win_row.get(),
                col_off: self.data_view.data_view_win_col.get(),
            },
        }
    }

    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    pub(super) fn render_data_frame(
        &self,
        frame: &mut ratatui::Frame,
        tensor: &TensorInfo,
        repr: Representation,
        slice: usize,
        view: ViewDtype,
        mode: SampleMode,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
        overlay: Option<&Overlay>,
    ) -> Result<(usize, bool, usize), String> {
        // Size the grid to the frame's render area — the live terminal size, or
        // the headless `TestBackend`'s fixed size, depending on the caller.
        let area = frame.area();
        let info = self.prepare_data_sample(
            tensor,
            repr,
            slice,
            view,
            mode,
            stats,
            area.width,
            area.height,
        )?;
        let cache = self.sample_cache.borrow();
        // Filled by the caller on the line before this one (the cache is what it just built).
        #[allow(clippy::unwrap_used)]
        let sample = &cache.as_ref().unwrap().sample;
        *self.clickable.borrow_mut() = match repr {
            Representation::Heatmap => UI::render_heatmap(frame, tensor, sample, stats),
            Representation::Values => UI::render_values(frame, tensor, sample, stats, stripe, base),
        };
        match overlay {
            Some(Overlay::Legend(l)) => UI::render_legend_band(frame, *l),
            Some(Overlay::Command(c)) => UI::render_command_band(frame, c),
            Some(Overlay::Notice(m)) => UI::render_notice_box(frame, m),
            None => {}
        }
        let badges = self.screen_badges(HelpCtx::Data);
        UI::render_badge_bar(frame, &badges, self.hovered_badge.get());
        self.links.borrow_mut().clear(); // data view shows no linkable names
        self.render_shortcut_hover(frame);
        Ok(info)
    }

    /// Render the data view from the *already sampled* result in
    /// [`Self::sample_cache`] (no re-sampling), with no overlay — used as the
    /// static background behind the reshape / slice prompts, which float over the
    /// view that was just drawn. A no-op if the cache is somehow empty.
    pub(super) fn render_cached_data(
        &self,
        frame: &mut ratatui::Frame,
        tensor: &TensorInfo,
        repr: Representation,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
    ) {
        let cache = self.sample_cache.borrow();
        let Some(cached) = cache.as_ref() else {
            return;
        };
        let sample = &cached.sample;
        // Drawn only as a static background behind a prompt (which runs its own
        // input loop), so the clickable map isn't updated here.
        let _regions = match repr {
            Representation::Heatmap => UI::render_heatmap(frame, tensor, sample, stats),
            Representation::Values => UI::render_values(frame, tensor, sample, stats, stripe, base),
        };
    }

    /// Render a tensor's data view to plain text via an in-memory Ratatui backend
    /// — the headless (`--plain`) data view and the `c` screen-copy share this.
    /// Mirrors [`Self::detail_plain`]. On a sampling error the message is rendered
    /// in place (matching the live "Data preview unavailable" path).
    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    pub(super) fn data_plain(
        &self,
        tensor: &TensorInfo,
        repr: Representation,
        slice: usize,
        view: ViewDtype,
        mode: SampleMode,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
        overlay: Option<&Overlay>,
    ) -> Result<String> {
        crate::tui::headless_render(120, 40, |f| {
            if let Err(msg) = self.render_data_frame(
                f, tensor, repr, slice, view, mode, stats, stripe, base, overlay,
            ) {
                use ratatui::widgets::{Paragraph, Widget};
                Paragraph::new(format!("Data preview unavailable: {msg}"))
                    .render(f.area(), f.buffer_mut());
            }
        })
    }
}
