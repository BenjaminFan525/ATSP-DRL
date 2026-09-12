# Portable environment regression fixtures

These are exact original bytes from two synthetic **training** cases generated
by the repository's HKBZ generator. The complete source paths, generator
metadata, file sizes, and SHA256 values are in `provenance.json`. No external
data, full benchmark dataset, tuning case, or blind evaluation case is included.

`regression_fixtures.case_dir` validates every indexed file before returning its
local test path. It does not fall back to an external dataset or the author's
workspace. These files are only regression inputs, not evidence of scientific
performance and not authorization to resume Stage2 training.

The action-mask, lookahead, and IGA contract tests retain original train case
0012; joint-policy tests retain original train case 0046. Departure-flow tests
previously loaded a separate tuning case but construct their own service and
departure states. They now use train case 0012 with all assertions unchanged.
