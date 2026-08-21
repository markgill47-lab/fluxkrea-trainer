"""The ``fk`` command line.

Today these commands call ``core`` directly, because there is no daemon
yet. At P3 the daemon arrives and the CLI becomes a full API client -
doc 06 is explicit that it is "a full API client, not a shortcut layer",
because every GUI action has to have a command or the fleet becomes
second-class.

The seam that makes that swap cheap is already here: each command parses
arguments, builds a request, and hands it to a *runner*. Only the runner
changes at P3.
"""
