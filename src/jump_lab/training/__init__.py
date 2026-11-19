# Import MjLab first. Its import runs the mjlab.tasks entry points, which import
# jump_lab.training.tasks; doing that from inside a half-imported jump_lab.training.mdp would fail.
import mjlab  # noqa: F401
