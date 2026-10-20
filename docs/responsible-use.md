# Responsible use



Watermark-remover is built for privacy, content hygiene, and research on files you own. It's not a tool for breaking copyright protections, passing off others' work as your own, or helping you evade detection when you're supposed to disclose AI use. The tool can remove marks, but removing a mark doesn't make stolen content yours, erase legal obligations, or protect you if you use this for something illegal.

## Legitimate Use

You can use this tool on:

- Your own AI-generated content you want cleaned for personal use or publication
- Files you own or have explicit permission to modify
- Research and educational work with institutional approval
- Security testing on systems you control
- Accessibility fixes on your own content

## Not for This

- Removing marks from content you don't own
- Claiming others' work as your own
- Bypassing copyright protections to redistribute material
- Evading AI detection to hide required disclosures
- Misrepresenting where content came from

If you use this tool, you're responsible for what you claim afterward. Removing a watermark is just a file operation. It doesn't grant you rights, erase your obligation to say "this is AI-generated" when you're supposed to, and it won't protect you from liability if you break the law.

## What We Can't Guarantee

The tool does best-effort removal. We don't promise that external detectors will miss what you removed, that an evaded detector won't catch you later, or that your cleaned content will pass any vendor's detection system. The README labels every method clearly: some cleaners are verifiable (they report exactly what was removed), others are best-effort (they might work, might not). Nothing here certifies detector failure.

## If You Use This

You agree to:

- Not use it to break protections on content you don't own
- Follow your jurisdiction's laws, including the DMCA (US), the DSM Directive (EU), and regional anti-circumvention rules
- Not use it for fraud, academic dishonesty, or copyright infringement
- Understand that removing a watermark doesn't grant ownership
- Disclose AI use when your employer, school, or publisher requires it

Not sure if your use case is legal? Talk to a lawyer or your institution first.

## On Legality

We can't give legal advice. Whether this is legal for you depends on what you own, where you live, what you're using it for, and which specific marks you're dealing with. Research often has exemptions; fraud and commercial infringement generally don't. If you have questions, consult a lawyer.

## Tell People You Used This

If you use watermark-remover in research papers or published work, consider disclosing it. Transparency builds trust and shows you're not trying to hide anything.

## Report Problems

Security concerns or misuse reports: see [SECURITY.md](https://github.com/PyModel/watermark-remover/blob/main/SECURITY.md).

## Primary references

- Anthropic, [*How Claude marks AI-generated content*](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4) (SynthID-Text, Nature 2024)
- Zhao et al., [*Invisible Image Watermarks Are Provably Removable Using Generative AI*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/10272bfd0371ef960ec557ed6c866058-Abstract-Conference.html) (NeurIPS 2024)
- [UnMarker](https://arxiv.org/abs/2405.08363) (IEEE S&P 2025)
- [CtrlRegen](https://arxiv.org/abs/2410.05470) (ICLR 2025)
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html)
- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://arxiv.org/abs/2301.10226)
