(function() {
    var projFile = app.project.file;
    if (!projFile) return;
    var jobDir = projFile.parent.parent;

    function findFile(folder, name, depth) {
        if (depth > 5) return null;
        var items = folder.getFiles();
        if (!items) return null;
        for (var f = 0; f < items.length; f++) {
            if (items[f] instanceof File && items[f].name === name) return items[f];
        }
        for (var f = 0; f < items.length; f++) {
            if (items[f] instanceof Folder) {
                var found = findFile(items[f], name, depth + 1);
                if (found) return found;
            }
        }
        return null;
    }

    // Map placeholder footage names to new image filenames
    var swaps = [
        {placeholder: "placeholder_1.png", newFile: "slide_0.png"},
        {placeholder: "placeholder_2.png", newFile: "slide_1.png"},
        {placeholder: "placeholder_3.png", newFile: "slide_2.png"}
    ];

    for (var s = 0; s < swaps.length; s++) {
        var swap = swaps[s];

        // Find the footage item
        var footageItem = null;
        for (var i = 1; i <= app.project.numItems; i++) {
            if (app.project.item(i).name === swap.placeholder) {
                footageItem = app.project.item(i);
                break;
            }
        }
        if (!footageItem || !footageItem.mainSource) continue;

        // Get the original file path that AE expects
        var originalPath = footageItem.mainSource.file;
        if (!originalPath) continue;

        // Find the new image in the job directory
        var newImage = findFile(jobDir, swap.newFile, 0);
        if (!newImage || !newImage.exists) continue;

        // Copy the new image OVER the original placeholder path
        // AE will then read the new content when rendering
        newImage.copy(originalPath);

        // Force AE to re-read the file
        footageItem.mainSource.reload();
    }
})();
