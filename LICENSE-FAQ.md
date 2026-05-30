# Grumpy license FAQ

Grumpy is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1) by Imaginary Biolabs GmbH. This FAQ explains common situations in plain language. It is not legal advice; when in doubt, read the [LICENSE](LICENSE) and <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">contact licensing</a>.

**Summary:** You may use Grumpy freely for development and for most internal production work. You may not offer Grumpy — or products built primarily around the restricted categories below — to third parties in commercial competition with Imaginary without a separate written license.

---

## When does Grumpy become open source?

Each version converts to **Apache License 2.0** on the **earlier** of:

1. **2030-12-31** (the Change Date in [LICENSE](LICENSE)), or  
2. The **fourth anniversary** of that version’s first public distribution under BSL.

So the first public release (for example on PyPI) may become Apache around **four years after launch**, even if that is before 2030-12-31.

---

## What production use is allowed without a commercial license?

The [Additional Use Grant](LICENSE) allows production use **except** when you offer the Licensed Work (or a substantial portion of it) to third parties as part of a product or service whose **primary purpose** is any of:

| Category | Examples (not exhaustive) |
|----------|---------------------------|
| **(i) Ragged array library** | A general-purpose numerical or columnar library for ragged/nested scientific data, distributed or hosted for others |
| **(ii) Bio-ML platform / registries** | Benchmark registry, model marketplace, dataset registry, or similar platform products |
| **(iii) Integrated biology environment** | An agent or environment that orchestrates biology tools as a product for others |
| **(iv) Fabric-like framework** | Composable ML pipelines on ragged data, YAML/JSON-driven datasets/benchmarks/metrics/transforms/models/layers, and an integrated biomolecular evaluation engine — including open-source or hosted alternatives to Imaginary’s **Fabric** product |

For **commercial entities**, those restrictions apply when the offering is in **commercial competition** with Imaginary or its platform products.

**Non-profit academic or research institutions** and **individuals doing personal non-commercial research** may use Grumpy in internal production **without** the commercial-competition test — but they still may **not** publish, distribute, or host products falling under (i)–(iv) for third parties without a written license.

---

## Common scenarios

### Can I use Grumpy in notebooks, coursework, and papers?

**Yes.** Non-production and internal production use for research is allowed. Cite Grumpy as you would any dependency.

### Can my university lab run HPC training pipelines on Grumpy internally?

**Yes**, if the lab is a non-profit academic or research institution and use stays **internal** (cluster jobs, shared lab storage, collaborators inside the project — not a product offered to outside customers).

### Can I publish my training code on GitHub?

**Usually yes**, if the repo is **application or experiment code** (a model, a benchmark run, analysis scripts) and not primarily a ragged-array library, registry, integrated biology agent, or Fabric-like framework offered to third parties.

If the repo’s main purpose is one of categories (i)–(iv), you need a <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">written license</a> before distributing it — including via public GitHub, PyPI, or a hosted service.

### Can I open-source a new ragged-array library built on Grumpy?

**Not without a written license** if that library is offered to third parties and its primary purpose is category **(i)** — even if you wrap or extend Grumpy rather than copying it wholesale.

For internal or non-production experimentation, BSL allows modification and redistribution subject to the license terms.

### Can my biotech company use Grumpy in drug-discovery pipelines?

**Yes for internal production**, as long as you are not **offering** to third parties a product or service whose primary purpose is (i)–(iv) in commercial competition with Imaginary.

Using Grumpy inside proprietary pipelines, models, and analyses is intended to be permitted.

### We are building a vertical SaaS (not a general ML platform). Is that OK?

**Often yes.** Restrictions target products whose **primary purpose** is a general ragged library, bio-ML platform/registry, integrated biology environment, or Fabric-like framework — not every application that happens to use arrays internally.

If your product’s main value is something else (assay analysis, a specific therapeutic program, a specialized design tool), you are likely outside (i)–(iv). Edge cases should be confirmed with <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">licensing</a>.

### What counts as “substantially similar to Fabric”?

**Fabric** (Imaginary’s ML pipeline layer on Grumpy) is described in category **(iv)**: standardized biomolecular data processing, training, and evaluation on ragged arrays, with composable transform pipelines and config-driven (YAML/JSON) definitions of datasets, benchmarks, metrics, transforms, models, or layers, plus an integrated evaluation engine — whether open source, hosted, or otherwise made available to third parties.

A thin wrapper around PyTorch/JAX for one benchmark is unlikely to qualify. A reusable, config-driven bio-ML stack meant for others to build on likely would.

### Can a public consortium (EBI-style) host a benchmark runner on Grumpy?

**Depends on primary purpose.** A public service whose main role is a **benchmark registry or Fabric-like evaluation platform** for third parties falls under (ii) or (iv) and requires a <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">written license</a>.

Internal or member-only infrastructure, or publishing **results** rather than the platform itself, may be fine — confirm for your deployment model.

### I’m spinning out a company from academia. What changes?

**Internal academic use** may have been fine under the carve-out. Once you **offer** a product or service to customers or the public, normal commercial rules apply — including the commercial-competition test and the (i)–(iv) categories.

Plan for a <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">commercial license</a> if your company’s product is platform-shaped.

### Can I fork Grumpy after the Change Date?

**Yes.** On the Change Date (or four-year anniversary, whichever is first) for a given version, that version is under **Apache 2.0**, including the right to use it in commercial products without the BSL Additional Use Grant restrictions.

---

## Commercial and partnership licenses

Imaginary offers written licenses for uses outside the Additional Use Grant — for example shipping a Grumpy-based library, a Fabric alternative, or a hosted bio-ML platform.

<a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">Contact licensing</a> with a short description of your organization, use case, and whether you plan to distribute or host software for third parties.

---

## Compliance

- You must retain the [LICENSE](LICENSE) text with copies and derivatives of Grumpy.
- Use in violation of the license terminates your rights automatically.

---

## Related documents

- [LICENSE](LICENSE) — full BSL 1.1 text and Additional Use Grant  
- [README](README.md) — project overview  
- [Imaginary Biolabs](https://www.imaginary.bio) — platform and products
