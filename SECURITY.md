# Security policy

Do not open a public issue for leaked credentials, unsafe deserialization, or a
vulnerability that could expose training infrastructure. Contact the repository
maintainer privately through the email listed in the corresponding paper.

Only load checkpoints from trusted sources. PyTorch pickle-based optimizer and
training-state files may execute code during deserialization; public releases
should distribute inference weights in `safetensors` format.
