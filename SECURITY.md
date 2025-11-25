# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security seriously in g2-forge. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Send an email to **brieuc@bdelaf.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (if available)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 7 days
- **Resolution**: We aim to provide a fix within 30 days for critical issues
- **Disclosure**: We will coordinate with you on public disclosure timing

### Scope

Security considerations for g2-forge include:

- **Code execution**: Arbitrary code execution via malicious inputs
- **Data integrity**: Corruption of training data or model weights
- **Dependency vulnerabilities**: Issues in third-party libraries
- **Model security**: Adversarial inputs affecting neural network behavior

### Out of Scope

The following are generally out of scope:

- Theoretical mathematical vulnerabilities in G2 geometry algorithms
- Performance issues (unless they enable denial of service)
- Issues requiring physical access to the machine
- Social engineering attacks

## Security Best Practices for Users

When using g2-forge:

1. **Validate inputs**: Always validate configuration files before loading
2. **Use virtual environments**: Isolate g2-forge dependencies
3. **Keep dependencies updated**: Regularly update PyTorch and other dependencies
4. **Review checkpoints**: Only load model checkpoints from trusted sources
5. **Secure credentials**: Never commit API keys or credentials to the repository

## Dependency Updates

We use GitHub's Dependabot to monitor and update dependencies. Critical security updates are prioritized and released as patch versions.

## Acknowledgments

We thank all security researchers who help keep g2-forge secure. Contributors who report valid security issues will be acknowledged (with permission) in our release notes.
