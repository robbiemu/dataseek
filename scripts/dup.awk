# find_dupes.awk (Corrected Version)
#
# DESCRIPTION:
# This script reads a log file line-by-line and prints groups of adjacent lines
# that have the same "core message".
#
# "Core message" is determined by heuristically removing prefixes like timestamps,
# log levels, and other metadata. The main strategy is to remove everything
# up to the last major delimiter (like ": " or " - ") on a line.

#
# USAGE:
# awk -f find_dupes.awk your_log_file.log
#

# This function extracts the core message from a full log line.
# 'core' is treated as a local variable by declaring it as an extra parameter.
function get_core(line, core) {
    # Start with the full line
    core = line

    # 1. Remove ANSI color codes, as they can interfere with pattern matching.
    gsub(/\x1b\[[0-9;]*m/, "", core)

    # 2. Greedily remove everything up to the last instance of " - ".
    # This is effective for standard log formats like "TIMESTAMP - LEVEL - message".
    if (match(core, /.*\s-\s/)) {
        sub(/.*\s-\s/, "", core)
    }

    # 3. Greedily remove everything up to the last instance of ": ".
    # This is effective for formats like "Component: Sub-component: message".
    if (match(core, /.*:\s/)) {
        sub(/.*:\s/, "", core)
    }

    # 4. Final cleanup: trim whitespace and remove common trailing characters.
    gsub(/^[ \t]+|[ \t]+$/, "", core)
    gsub(/\.\.\.$/, "", core)
    gsub(/['")]$/, "", core)

    return core
}

# This main block is executed for each line of the input file.
{
    # Get the core message of the current line.
    current_core = get_core($0)

    # Compare with the previous line's core message.
    # NR > 1 ensures we skip the very first line.
    # We also ignore blank or empty core messages.
    if (NR > 1 && current_core != "" && current_core == prev_core) {
        # This line is a duplicate of the one before it.

        if (in_duplicate_block == 0) {
            # This is the start of a new block of duplicates.
            # Print a header and the first line of the group (the previous line).
            print "--- Duplicate Group ---"
            print prev_original
            in_duplicate_block = 1
        }
        # Print the current duplicate line.
        print $0
    } else {
        # This line is different from the previous one.

        if (in_duplicate_block == 1) {
            # If we were just in a duplicate block, print the closing footer.
            print "-----------------------"
        }
        in_duplicate_block = 0
    }

    # Store the current line's original text and its core message for the next loop iteration.
    prev_original = $0
    prev_core = current_core
}

# This END block is executed once after all lines have been read.
END {
    if (in_duplicate_block == 1) {
        # If the file ends while inside a duplicate block, print the closing footer.
        print "-----------------------"
    }
}
