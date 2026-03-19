/*
 * The Docket — Cinematic Reel Template Builder v2
 * Run in After Effects: File > Scripts > Run Script File
 *
 * Creates a 1080×1920 Reels template with:
 *   - Title → Hook → Body 1-3 → Closing → CTA slide structure
 *   - Per-slide image placeholders with Ken Burns (scale + drift)
 *   - Dark overlay on each image for text legibility
 *   - Subtitle-style text at lower third (cinematic) or centered (narrative)
 *   - Crossfade transitions between slides
 *   - One glitch micro-cut (6 frames) on a random transition
 *   - Voiceover + background music audio placeholders
 *   - Section color theming via accent color layer
 *   - Title card with bold ALL CAPS + vignette
 *
 * Text wrapping handled by Python job builder (inserts \n before submission).
 * Nexrender injects: text via "data" assets, images via "image" assets,
 * audio via "audio" assets.
 */

(function () {
    // ── Dimensions & Timing ──────────────────────────────────────────
    var W = 1080, H = 1920, FPS = 30;
    var FADE = 15; // frames (0.5s)

    // Slide durations (defaults — voiceover timing overrides via Nexrender)
    var TITLE_DUR   = 3;
    var HOOK_DUR    = 4;
    var SLIDE_DUR   = 5;
    var CLOSING_DUR = 4;
    var CTA_DUR     = 3;
    var DURATION    = TITLE_DUR + HOOK_DUR + (SLIDE_DUR * 3) + CLOSING_DUR + CTA_DUR; // 29s

    // ── Colors ───────────────────────────────────────────────────────
    var BG_DARK   = [0.07, 0.07, 0.07];       // #121212
    var WHITE     = [1, 1, 1];
    var ACCENT    = [78/255, 205/255, 196/255]; // #4ecdc4 (default teal)
    var OVERLAY_OPACITY = 55; // % darkness over images

    // ── Layout ───────────────────────────────────────────────────────
    var MARGIN = 80;
    var SUBTITLE_Y = H * 0.82;  // lower-third subtitle position
    var CENTER_Y   = H * 0.45;  // centered text position
    var TITLE_Y    = H * 0.42;  // title position

    // ── Project & Comp ───────────────────────────────────────────────
    var proj = app.newProject();
    var comp = proj.items.addComp("main", W, H, 1, DURATION, FPS);

    // ── Glitch pre-comp ──────────────────────────────────────────────
    // 6-frame (0.2s) glitch effect: RGB channel offset + scale jitter
    var GLITCH_DUR = 6 / FPS;
    var glitchComp = proj.items.addComp("glitch_fx", W, H, 1, GLITCH_DUR, FPS);

    // Red channel layer (shifted left)
    var glitchR = glitchComp.layers.addSolid([1, 0, 0], "glitch_r", W, H, 1);
    glitchR.property("Transform").property("Position").setValue([W/2 - 12, H/2]);
    glitchR.property("Transform").property("Opacity").setValue(33);
    glitchR.blendingMode = BlendingMode.ADD;

    // Blue channel layer (shifted right)
    var glitchB = glitchComp.layers.addSolid([0, 0, 1], "glitch_b", W, H, 1);
    glitchB.property("Transform").property("Position").setValue([W/2 + 12, H/2]);
    glitchB.property("Transform").property("Opacity").setValue(33);
    glitchB.blendingMode = BlendingMode.ADD;

    // White flash frame (frame 1 and 4 only)
    var flashSolid = glitchComp.layers.addSolid([1, 1, 1], "flash", W, H, 1);
    var flashOp = flashSolid.property("Transform").property("Opacity");
    flashOp.setValueAtTime(0, 80);
    flashOp.setValueAtTime(1/FPS, 0);
    flashOp.setValueAtTime(3/FPS, 0);
    flashOp.setValueAtTime(4/FPS, 60);
    flashOp.setValueAtTime(5/FPS, 0);

    // Scale jitter on a null
    var jitterNull = glitchComp.layers.addNull();
    jitterNull.name = "jitter";
    var jScale = jitterNull.property("Transform").property("Scale");
    jScale.setValueAtTime(0, [102, 98]);
    jScale.setValueAtTime(2/FPS, [98, 103]);
    jScale.setValueAtTime(4/FPS, [101, 97]);
    jScale.setValueAtTime(GLITCH_DUR, [100, 100]);

    // ── Helpers ──────────────────────────────────────────────────────

    // Add a text layer with fade in/out
    function addText(name, placeholder, yPos, fontSize, color, startTime, dur, isBold) {
        var displayText = isBold ? placeholder.toUpperCase() : placeholder;
        var layer = comp.layers.addText(displayText);
        layer.name = name;
        layer.startTime = startTime;
        layer.outPoint = startTime + dur;

        var srcText = layer.property("ADBE Text Properties").property("ADBE Text Document");
        var doc = srcText.value;
        doc.resetCharStyle();
        doc.fontSize = fontSize;
        doc.fillColor = color;
        doc.font = isBold ? "Arial-BoldMT" : "ArialMT";
        doc.justification = ParagraphJustification.CENTER_JUSTIFY;
        srcText.setValue(doc);

        layer.property("Transform").property("Position").setValue([W / 2, yPos]);

        // Fade in/out
        var opacity = layer.property("Transform").property("Opacity");
        opacity.setValueAtTime(startTime, 0);
        opacity.setValueAtTime(startTime + FADE/FPS, 100);
        opacity.setValueAtTime(startTime + dur - FADE/FPS, 100);
        opacity.setValueAtTime(startTime + dur, 0);

        return layer;
    }

    // Import individual placeholder PNGs so each layer has its own footage source.
    // Nexrender replaces each source independently via replaceSource().
    var desktopPath = Folder.desktop.fsName;
    var placeholderNames = ["title", "hook", "1", "2", "3", "closing"];
    var placeholderItems = {};
    for (var p = 0; p < placeholderNames.length; p++) {
        var pName = placeholderNames[p];
        var pFile = new File(desktopPath + "/placeholder_" + pName + ".png");
        if (pFile.exists) {
            var pIO = new ImportOptions(pFile);
            pIO.importAs = ImportAsType.FOOTAGE;
            placeholderItems[pName] = proj.importFile(pIO);
        }
    }

    // Add an image placeholder (footage) with Ken Burns animation
    // Nexrender replaces the footage source with the AI-generated image
    function addImageBg(name, startTime, dur) {
        // Extract suffix from name (e.g. "bg_image_1" → "1", "bg_image_title" → "title")
        var suffix = name.replace("bg_image_", "");
        var layer;
        if (placeholderItems[suffix]) {
            layer = comp.layers.add(placeholderItems[suffix]);
            layer.property("Transform").property("Scale").setValue([100, 100]);
        } else {
            layer = comp.layers.addSolid(BG_DARK, name, W, H, 1);
        }
        layer.name = name;
        layer.startTime = startTime;
        layer.outPoint = startTime + dur;

        // Ken Burns: slow zoom from 105% → 100% with slight Y drift
        var scale = layer.property("Transform").property("Scale");
        scale.setValueAtTime(startTime, [105, 105]);
        scale.setValueAtTime(startTime + dur, [100, 100]);

        var pos = layer.property("Transform").property("Position");
        pos.setValueAtTime(startTime, [W/2, H/2 - 15]);
        pos.setValueAtTime(startTime + dur, [W/2, H/2 + 15]);

        // Crossfade opacity
        var opacity = layer.property("Transform").property("Opacity");
        opacity.setValueAtTime(startTime, 0);
        opacity.setValueAtTime(startTime + FADE/FPS, 100);
        opacity.setValueAtTime(startTime + dur - FADE/FPS, 100);
        opacity.setValueAtTime(startTime + dur, 0);

        return layer;
    }

    // Add dark overlay on top of an image layer
    function addDarkOverlay(name, startTime, dur) {
        var layer = comp.layers.addSolid([0, 0, 0], name, W, H, 1);
        layer.startTime = startTime;
        layer.outPoint = startTime + dur;
        layer.property("Transform").property("Opacity").setValue(OVERLAY_OPACITY);
        return layer;
    }

    // Add vignette — simple solid with low opacity, no mask needed
    // The dark overlay already handles legibility; this just darkens edges slightly
    function addVignette(name, startTime, dur) {
        var layer = comp.layers.addSolid([0, 0, 0], name, W, H, 1);
        layer.name = name;
        layer.startTime = startTime;
        layer.outPoint = startTime + dur;
        // Scale up so edges are darker than center (poor man's vignette)
        layer.property("Transform").property("Scale").setValue([120, 120]);
        layer.property("Transform").property("Opacity").setValue(30);
        return layer;
    }

    // ── Build Timeline ───────────────────────────────────────────────
    var t = 0;
    var transitionTimes = []; // collect transition timestamps for glitch placement

    // --- Logo bug (persistent across entire video) ---
    var logoBug = addText("logo_bug", "THE DOCKET", 120, 24, ACCENT, 0, DURATION, true);

    // --- Slide 1: TITLE ---
    addImageBg("bg_image_title", t, TITLE_DUR);
    addDarkOverlay("overlay_title", t, TITLE_DUR);
    addVignette("vignette_title", t, TITLE_DUR);
    addText("title_text", "TITLE TEXT", TITLE_Y, 72, WHITE, t, TITLE_DUR, true);
    t += TITLE_DUR;
    transitionTimes.push(t);

    // --- Slide 2: HOOK ---
    addImageBg("bg_image_hook", t, HOOK_DUR);
    addDarkOverlay("overlay_hook", t, HOOK_DUR);
    addText("hook_text", "HOOK TEXT", SUBTITLE_Y, 42, WHITE, t, HOOK_DUR, false);
    t += HOOK_DUR;
    transitionTimes.push(t);

    // --- Slides 3-5: BODY ---
    for (var i = 1; i <= 3; i++) {
        addImageBg("bg_image_" + i, t, SLIDE_DUR);
        addDarkOverlay("overlay_" + i, t, SLIDE_DUR);
        addText("body_text_" + i, "SLIDE " + i, SUBTITLE_Y, 42, WHITE, t, SLIDE_DUR, false);
        t += SLIDE_DUR;
        if (i < 3) transitionTimes.push(t); // don't add after last body slide
    }
    transitionTimes.push(t); // body→closing transition

    // --- Slide 6: CLOSING ---
    addImageBg("bg_image_closing", t, CLOSING_DUR);
    addDarkOverlay("overlay_closing", t, CLOSING_DUR);
    addText("closing_text", "CLOSING TEXT", CENTER_Y, 54, ACCENT, t, CLOSING_DUR, false);
    t += CLOSING_DUR;

    // --- Slide 7: CTA ---
    var ctaBg = comp.layers.addSolid(BG_DARK, "bg_cta", W, H, 1);
    ctaBg.startTime = t;
    ctaBg.outPoint = t + CTA_DUR;
    addText("cta_text", "Follow for more stories\nlike this at The Docket", H * 0.40, 48, WHITE, t, CTA_DUR, false);
    addText("cta_handle", "@docketclimate", H * 0.55, 36, ACCENT, t, CTA_DUR, false);

    // ── Glitch micro-cut ─────────────────────────────────────────────
    // Place the glitch pre-comp at one random transition point.
    // Randomize using current time as seed.
    var glitchIdx = Math.floor(Math.random() * transitionTimes.length);
    var glitchTime = transitionTimes[glitchIdx] - (GLITCH_DUR / 2); // center on transition

    var glitchLayer = comp.layers.add(glitchComp);
    glitchLayer.name = "glitch_cut";
    glitchLayer.startTime = glitchTime;
    glitchLayer.outPoint = glitchTime + GLITCH_DUR;
    glitchLayer.blendingMode = BlendingMode.ADD;
    // Move to top of layer stack
    glitchLayer.moveToBeginning();

    // ── Audio placeholder layers ─────────────────────────────────────
    var voNull = comp.layers.addNull();
    voNull.name = "voiceover_audio";
    voNull.startTime = 0;
    voNull.outPoint = DURATION;
    voNull.enabled = false;

    var musicNull = comp.layers.addNull();
    musicNull.name = "bg_music";
    musicNull.startTime = 0;
    musicNull.outPoint = DURATION;
    musicNull.enabled = false;

    // ── Section accent color reference ───────────────────────────────
    // Nexrender can update this null's effects to change the accent color
    var accentNull = comp.layers.addNull();
    accentNull.name = "accent_color";
    accentNull.enabled = false;

    // ── Save ─────────────────────────────────────────────────────────
    var savePath = Folder.desktop.fsName + "/docket_cinematic.aep";
    proj.save(new File(savePath));

    alert(
        "Cinematic template v2 built!\n\n" +
        "Slides: Title > Hook > Body 1-3 > Closing > CTA\n" +
        "Ken Burns on all image layers\n" +
        "Dark overlay + vignette on title\n" +
        "Glitch micro-cut at transition " + (glitchIdx + 1) + " of " + transitionTimes.length + "\n" +
        "Handle: @docketclimate\n\n" +
        "Saved to Desktop."
    );
})();
