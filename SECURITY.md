# Security Policy

This document defines security procedures, vulnerability reporting, and secrets handling rules.

## Supported Versions

Only the latest release version on the main branch receives security updates.

| Version | Supported |
|---------|-----------|
| Latest  | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability, please do not post it in a public issue.

Submit reports privately:
- Email: manikantasuryasai21295cm055@gmail.com
- Subject: [SECURITY] <project-name> — <Brief description>

Include the following details in your report:
1. Steps to reproduce the issue.
2. The potential impact of the vulnerability.
3. Suggested patches or mitigations if available.

We will acknowledge your report within 48 hours and discuss remediation timelines with you.

## Security Best Practices

- **Secrets Handling:** Never commit secrets, credentials, API keys, or certificates. Inject secrets via environment variables.
- **Dependency Checks:** We use automated tools to scan dependencies for known vulnerabilities and apply updates regularly.
- **Input Validation:** Ensure user inputs are sanitized before processing to prevent scripting or injection issues.
- **Authentication Safety:** Verify credentials are not exposed in client logs or client-facing components.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)