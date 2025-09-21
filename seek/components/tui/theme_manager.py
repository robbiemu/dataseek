from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import darkdetect

    DARKDETECT_AVAILABLE = True
except ImportError:
    DARKDETECT_AVAILABLE = False

from textual.widgets import Footer

if TYPE_CHECKING:
    from .dataseek_tui import DataSeekTUI


def detect_system_theme(app: DataSeekTUI) -> str | None:
    """Detect the current system theme (dark/light)."""
    if not DARKDETECT_AVAILABLE:
        return None

    try:
        # darkdetect.isDark() returns True for dark mode, False for light mode, None if unknown
        is_dark = darkdetect.isDark()
        if is_dark is None:
            return None
        return "dark" if is_dark else "light"
    except Exception as e:
        app.debug_log(f"Failed to detect system theme: {e}")
        return None


def sync_with_system_theme(app: DataSeekTUI) -> None:
    """Sync the app theme with the system theme if it has changed."""
    if not app.system_theme_sync_enabled:
        return

    current_system_theme = detect_system_theme(app)
    if current_system_theme is None:
        return

    # Only change if system theme is different from what we detected last time
    if current_system_theme != app.last_detected_theme:
        # If the user manually toggled, clear the override only when the
        # system theme actually changes. This avoids immediate auto-revert.
        if app.user_theme_override:
            app.user_theme_override = False
        app.last_detected_theme = current_system_theme

        # Convert system theme to our dark boolean
        system_wants_dark = current_system_theme == "dark"

        # Only apply if different from current app theme
        if system_wants_dark != app.dark:
            app.dark = system_wants_dark
            theme_name = "dark" if app.dark else "light"

            # Apply theme styling
            apply_theme_styling(app)

            # Show user feedback
            if hasattr(app, "conversation"):
                app.conversation.add_message("info", f"🌓 Auto-synced to system {theme_name} theme")

            app.debug_log(f"Auto-synced theme to system: {theme_name}")


def apply_footer_styles(app: DataSeekTUI) -> None:
    """Applies custom styles to the footer keys based on the current theme."""

    def _apply_styles() -> None:
        try:
            footer = app.query_one(Footer)
            key_color = "#58a6ff" if app.dark else "#0969da"

            # Try multiple selectors for footer keys
            selectors_tried = []
            keys_found = 0

            for selector in [".footer-key", "FooterKey", "Key", ".key"]:
                try:
                    keys = footer.query(selector)
                    selectors_tried.append(f"{selector}:{len(keys)}")
                    if keys:
                        keys_found += len(keys)
                        for key in keys:
                            key.styles.color = key_color
                except Exception as e:
                    selectors_tried.append(f"{selector}:error({e})")

            # Also try CSS class approach for footer keys
            theme_class = "dark-theme" if app.dark else "light-theme"
            opposite_class = "light-theme" if app.dark else "dark-theme"

            try:
                for selector in ["FooterKey", ".footer-key"]:
                    for key in footer.query(selector):
                        key.add_class(theme_class)
                        key.remove_class(opposite_class)
            except Exception:
                pass  # nosec B110 # - non-critical visual update failure

            app.debug_log(
                f"Footer key styling - selectors tried: {selectors_tried}, keys found: {keys_found}, color: {key_color}"
            )

        except Exception as e:
            app.debug_log(f"Footer styling failed: {e}")

    # Schedule the styling to happen after the footer is fully rendered
    app.call_after_refresh(_apply_styles)


def apply_theme_styling(app: DataSeekTUI) -> None:
    """Apply theme styling to all elements programmatically."""

    def _apply_theme_styles() -> None:
        try:
            # Define colors for current theme
            if app.dark:
                bg_color = "#0d1117"
                text_color = "#c9d1d9"
                header_bg = "#21262d"
                container_bg = "#161b22"
                conversation_bg = "#1b1f27"
                progress_bg = "#1b1f27"
                mission_bg = "#1f2230"
                title_color = "#58a6ff"
                progress_text_color = "#58a6ff"
            else:
                bg_color = "#f6f8fa"
                text_color = "#24292f"
                header_bg = "#eaeef2"
                container_bg = "#ffffff"
                conversation_bg = "#f0f2f6"
                progress_bg = "#f0f2f6"
                mission_bg = "#f6f8fa"
                title_color = "#0969da"
                progress_text_color = "#0969da"

            theme_name = "dark" if app.dark else "light"
            app.debug_log(f"Applying {theme_name} theme styling...")

            # Apply screen/app background if possible
            try:
                app.screen.styles.background = bg_color
                app.screen.styles.color = text_color
            except Exception:
                pass  # nosec B110 # - non-critical visual update failure

            # Apply built-in Header widget styling with multiple approaches
            try:
                from textual.widgets import Header

                header_widget = app.query_one(Header)

                # Try direct styling
                header_widget.styles.background = header_bg
                header_widget.styles.color = text_color

                # Try forcing a refresh
                header_widget.refresh()

                # Try setting CSS variables if they exist
                try:
                    header_widget.styles.background = header_bg
                    header_widget.styles.color = text_color
                except Exception as e:
                    app.debug_log(f"Failed removing stats_header class: {e}")

                # Try accessing sub-components of the header
                try:
                    for child in header_widget.children:
                        child.styles.background = header_bg
                        child.styles.color = text_color
                except Exception:
                    pass  # nosec B110 # - non-critical visual update failure

                # Try adding/removing CSS classes for theme control
                try:
                    if app.dark:
                        header_widget.add_class("dark-theme")
                        header_widget.remove_class("light-theme")
                    else:
                        header_widget.add_class("light-theme")
                        header_widget.remove_class("dark-theme")
                except Exception:
                    pass  # nosec B110 # - non-critical visual update failure

                app.debug_log(f"Applied Header widget styling: bg={header_bg}, color={text_color}")
            except Exception as e:
                app.debug_log(f"Header widget styling failed: {e}")

            # Apply stats header styling (custom widget)
            try:
                # Try both the widget reference and ID selector
                if hasattr(app, "stats_header"):
                    # Apply background and text color
                    app.stats_header.styles.background = header_bg
                    app.stats_header.styles.color = text_color
                    # Force refresh to pick up the new colors
                    app.stats_header.refresh()
                    app.debug_log("Applied stats_header reference styling")

                # Also try targeting by ID
                try:
                    stats_header = app.query_one("#stats-header")
                    stats_header.styles.background = header_bg
                    stats_header.styles.color = text_color
                    stats_header.refresh()
                    app.debug_log("Applied stats_header ID styling")
                except Exception:
                    pass  # nosec B110 # - non-critical visual update failure

                # Try adding CSS classes for theme control
                try:
                    if hasattr(app, "stats_header"):
                        if app.dark:
                            app.stats_header.add_class("dark-theme")
                            app.stats_header.remove_class("light-theme")
                        else:
                            app.stats_header.add_class("light-theme")
                            app.stats_header.remove_class("dark-theme")
                except Exception:
                    pass  # nosec B110 # - non-critical visual update failure

            except Exception as e:
                app.debug_log(f"Stats header styling failed: {e}")

            # Apply main container styling
            try:
                container = app.query_one("#main-container")
                container.styles.background = container_bg
                container.styles.color = text_color
            except Exception as e:
                app.debug_log(f"Failed styling main container: {e}")

            # Apply progress panel styling
            if hasattr(app, "progress_panel"):
                try:
                    app.progress_panel.styles.background = progress_bg
                    app.progress_panel.styles.color = text_color
                except Exception as e:
                    app.debug_log(f"Failed styling progress_panel: {e}")

            # Apply mission panel styling
            if hasattr(app, "mission_panel"):
                try:
                    app.mission_panel.styles.background = mission_bg
                    app.mission_panel.styles.color = text_color
                except Exception as e:
                    app.debug_log(f"Failed styling mission_panel: {e}")

            # Apply conversation styling
            if hasattr(app, "conversation"):
                try:
                    app.conversation.styles.background = conversation_bg
                    app.conversation.styles.color = text_color
                except Exception as e:
                    app.debug_log(f"Failed styling conversation panel: {e}")

            # Apply title styling
            for title in app.query(".panel-title"):
                try:
                    title.styles.color = title_color
                except Exception as e:
                    app.debug_log(f"Failed styling header title: {e}")

            # Apply progress text styling
            try:
                progress = app.query_one("#main-progress")
                progress.styles.color = progress_text_color
            except Exception as e:
                app.debug_log(f"Failed styling footer progress: {e}")

            # Apply CSS classes to all elements for theme control
            try:
                # Apply theme class to all major elements
                theme_class = "dark-theme" if app.dark else "light-theme"
                opposite_class = "light-theme" if app.dark else "dark-theme"

                # List of all elements to apply theme classes to
                element_selectors = [
                    "#progress-panel",
                    "#main-progress",
                    "#recent-samples",
                    "#mission-panel",
                    "#mission-info",
                    "#conversation",
                    "#main-container",
                ]

                for selector in element_selectors:
                    try:
                        element = app.query_one(selector)
                        element.add_class(theme_class)
                        element.remove_class(opposite_class)
                    except Exception as e:
                        app.debug_log(f"Failed updating title classes: {e}")

                # Apply theme class to all panel titles
                for title in app.query(".panel-title"):
                    try:
                        title.add_class(theme_class)
                        title.remove_class(opposite_class)
                    except Exception as e:
                        app.debug_log(f"Failed updating scrollbar class via {selector}: {e}")

                app.debug_log(f"Applied CSS theme classes: {theme_class}")

            except Exception as e:
                app.debug_log(f"CSS class application failed: {e}")

            # Apply scrollbar styling with multiple approaches
            try:
                theme_class = "dark-theme" if app.dark else "light-theme"
                opposite_class = "light-theme" if app.dark else "dark-theme"

                # Try multiple scrollbar element selectors
                scrollbar_selectors = [
                    ".scrollbar",
                    "Scrollbar",
                    ".vertical-scrollbar",
                    ".horizontal-scrollbar",
                ]

                thumb_selectors = [".scrollbar-thumb", "ScrollbarThumb", ".thumb"]

                # Apply to all scrollbar elements in the app
                for selector in scrollbar_selectors:
                    try:
                        for scrollbar in app.query(selector):
                            scrollbar.add_class(theme_class)
                            scrollbar.remove_class(opposite_class)
                    except Exception as e:
                        app.debug_log(f"Failed updating thumb class via {selector}: {e}")

                # Apply to all scrollbar thumb elements
                for selector in thumb_selectors:
                    try:
                        for thumb in app.query(selector):
                            thumb.add_class(theme_class)
                            thumb.remove_class(opposite_class)
                    except Exception as e:
                        app.debug_log(f"Failed querying elements for selector {selector}: {e}")

                # Debug: Log what scrollbar elements we can find
                found_elements = []
                for selector in scrollbar_selectors + thumb_selectors:
                    try:
                        elements = app.query(selector)
                        if elements:
                            found_elements.append(f"{selector}: {len(elements)} elements")
                    except Exception as e:
                        app.debug_log(f"Failed enumerating all elements: {e}")

                if found_elements:
                    app.debug_log(f"Found scrollbar elements: {', '.join(found_elements)}")
                else:
                    app.debug_log("No scrollbar elements found with any selector")

                    # Try to find ALL elements and log them
                    all_elements = []
                    try:
                        for element in app.query("*"):
                            if hasattr(element, "id") and element.id:
                                all_elements.append(element.id)
                            elif hasattr(element, "__class__"):
                                class_name = element.__class__.__name__
                                if "scroll" in class_name.lower():
                                    all_elements.append(class_name)

                        if all_elements:
                            app.debug_log(
                                f"All elements with IDs or scroll-related: {', '.join(set(all_elements))}"
                            )
                    except Exception as e:
                        app.debug_log(f"Failed to log found elements: {e}")

            except Exception as e:
                app.debug_log(f"Scrollbar styling failed: {e}")

        except Exception as e:
            app.debug_log(f"Theme styling error: {e}")

    # Apply built-in Footer widget styling
    def _apply_footer_widget_styles() -> None:
        try:
            from textual.widgets import Footer

            footer_widget = app.query_one(Footer)
            # Use the same color variables from the parent scope
            current_footer_bg = "#21262d" if app.dark else "#eaeef2"
            current_footer_text = "#c9d1d9" if app.dark else "#24292f"
            footer_widget.styles.background = current_footer_bg
            footer_widget.styles.color = current_footer_text
            app.debug_log(
                f"Applied Footer widget styling: bg={current_footer_bg}, color={current_footer_text}"
            )
        except Exception as e:
            app.debug_log(f"Footer widget styling failed: {e}")

    app.call_after_refresh(_apply_footer_widget_styles)

    # Apply footer key styles
    apply_footer_styles(app)

    # Apply main theme styles
    app.call_after_refresh(_apply_theme_styles)


def action_toggle_theme(app: DataSeekTUI) -> None:
    """An action to toggle between dark and light themes (manual override)."""
    # Toggle theme manually
    app.dark = not app.dark
    theme_name = "dark" if app.dark else "light"

    # Mark a manual override and align last_detected_theme to current
    # system theme so periodic checks don't immediately revert it.
    if app.system_theme_sync_enabled:
        app.user_theme_override = True
        current_system = detect_system_theme(app)
        if current_system is not None:
            app.last_detected_theme = current_system

    # Apply all theme styling programmatically
    apply_theme_styling(app)

    # Show user feedback
    if hasattr(app, "conversation"):
        sync_status = " (manual override)" if app.system_theme_sync_enabled else ""
        app.conversation.add_message("info", f"🎨 Switched to {theme_name} theme{sync_status}")

    app.debug_log(f"Theme manually toggled to: {theme_name}")
