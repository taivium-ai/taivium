# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Taivium, please **do not** open a public GitHub issue. Instead, please report it confidentially:

### Coordinated Disclosure

1. **Email**: security@taivium.dev
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce (if applicable)
   - Impact assessment
   - Suggested fix (if available)

3. **Timeline**:
   - We will acknowledge receipt within 48 hours
   - We will provide a status update within 7 days
   - We aim to release a fix within 30 days of confirmation
   - We will credit you in the security advisory (unless you prefer anonymity)

### Public Disclosure

Once a fix is released, we will:
- Publish a security advisory on GitHub
- Credit the researcher (with permission)
- Include CVE information if applicable

---

## Security Best Practices for Users

### Installation
```bash
# Always use a pinned version
pip install taivium==0.2.0

# Or use a version constraint
pip install "taivium>=0.2.0,<0.2.0"
```

### Session Store
- **In-Memory**: Safe for single-process applications only
- **Redis**: Always use TLS/SSL in production:
  ```python
  from taivium import Taivium
  
  pipeline = Taivium(
      session_store_type="redis",
      session_store_kwargs={
          "host": "redis.example.com",
          "port": 6380,
          "ssl": True,
          "ssl_cert_reqs": "required"
      }
  )
  ```

### API Key Management
- Store API keys in environment variables, not code:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- Use `.env` files with `.gitignore` protection:
  ```bash
  echo ".env" >> .gitignore
  ```
- Rotate API keys regularly

### PII Handling
- Taivium detects and anonymizes sensitive data
- Verified with 99.8% test pass rate across 467 tests
- Always review detection results in development
- Use audit logs to track anonymizations

---

## Dependency Management

We monitor dependencies for security vulnerabilities using:
- **pip-audit**: Detects known CVEs in installed packages
- **Bandit**: Scans source code for common security issues
- **detect-secrets**: Prevents accidental secret commits
- **GitHub Dependabot**: Automatic security update PRs

### Vulnerability Tracking
- Check [GitHub Security Advisories](https://github.com/taivium-ai/taivium/security/advisories)
- Subscribe to [GitHub Notifications](https://github.com/taivium-ai/taivium/watchers)

---

## Security Audit Results

Last audit: **2026-06-06**

### Dependency Scan
- ✅ No high-severity vulnerabilities
- ⚠️ 3 known issues (pip, transformers)
  - Action: Automated dependency updates via Dependabot

### Code Security (SAST)
- ✅ No critical issues
- ⚠️ 2 Low severity false positives (label mappings)
- ⚠️ 1 Medium: Missing revision pin on HuggingFace downloads
  - Status: Fixed in next release

### Secrets Detection
- ✅ No real secrets found
- ⚠️ 1 false positive ("API_KEY" string in detection label)

### License Audit
- ✅ MIT-compatible licenses
- ⚠️ Indirect LGPLv2+ dependencies (chardet)
  - Status: Acceptable under MIT license (weak copyleft)

---

## Compliance

Taivium is designed for:
- **GDPR**: Sensitive data detection and pseudonymization
- **CCPA**: PII identification and removal
- **HIPAA**: Healthcare PII (patient names, IDs, medical record numbers)
- **PCI-DSS**: Credit card and financial data detection

---

## Contact

- **Security Issues**: security@taivium.dev
- **General Support**: support@taivium.dev
- **GitHub Issues**: [Issues](https://github.com/taivium-ai/taivium/issues)
