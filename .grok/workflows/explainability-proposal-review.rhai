let meta = #{
    name: "explainability-proposal-review",
    description: "Multi-agent review of the single-image explainability proposal against LV code and U-Net process",
    when_to_use: "Review docs/single-image-explainability-proposal.md from product, architecture, code-fact, explainability, and process angles",
    phases: [
        #{ title: "Specialists", detail: "five independent reviews of the proposal vs code" },
        #{ title: "Synthesis", detail: "one staff memo with decisions and module split" },
    ],
};

let review_schema = #{
    "type": "object",
    "required": ["role", "verdict", "summary", "findings", "open_questions", "recommendations"],
    "properties": #{
        "role": #{ "type": "string" },
        "verdict": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "findings": #{
            "type": "array",
            "maxItems": 10,
            "items": #{
                "type": "object",
                "required": ["severity", "title", "evidence", "implication"],
                "properties": #{
                    "severity": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "evidence": #{ "type": "string" },
                    "implication": #{ "type": "string" },
                },
            },
        },
        "open_questions": #{
            "type": "array",
            "maxItems": 8,
            "items": #{ "type": "string" },
        },
        "recommendations": #{
            "type": "array",
            "maxItems": 8,
            "items": #{ "type": "string" },
        },
    },
};

let memo_schema = #{
    "type": "object",
    "required": ["headline", "overall_verdict", "what_to_keep", "what_to_change", "blockers", "product_decisions", "suggested_unet_split", "memo_markdown"],
    "properties": #{
        "headline": #{ "type": "string" },
        "overall_verdict": #{ "type": "string" },
        "what_to_keep": #{ "type": "array", "items": #{ "type": "string" } },
        "what_to_change": #{ "type": "array", "items": #{ "type": "string" } },
        "blockers": #{ "type": "array", "items": #{ "type": "string" } },
        "product_decisions": #{ "type": "array", "items": #{ "type": "string" } },
        "suggested_unet_split": #{ "type": "array", "items": #{ "type": "string" } },
        "memo_markdown": #{ "type": "string" },
    },
};

let doc_path = "docs/single-image-explainability-proposal.md";
if args != () {
    if args.doc_path != () {
        doc_path = args.doc_path;
    }
}

fn specialist_prompt(role, focus, doc_path) {
    let p = "You are reviewing a VisualLatent (LV) product/architecture proposal. ";
    p += "This is a mature Streamlit dataset analysis tool. New features follow a U-Net spec overlay: ";
    p += "1_user_needs (feature) -> 2_PO_PRD + ROADMAP (split feature into modules) -> 3_Architect_Design (design+AC, no code) -> tests/ (PM) -> scripts/ (PG). ";
    p += "Implementation goes into scripts/ and tests/, not a sandbox. Never write user datasets; cache goes to .lv_cache/. ";
    p += "The proposal is NOT yet a user_needs or PRD. Your job is to think, not implement. Do not edit files. ";
    p += "\n\nYOUR ROLE: ";
    p += role;
    p += "\nFOCUS: ";
    p += focus;
    p += "\n\nREQUIRED READING (use read_file / grep / list_dir; do not answer from memory): ";
    p += "1) ";
    p += doc_path;
    p += "  — the proposal under review. ";
    p += "2) ROADMAP.md (current milestone state; new capability must start a new /user round). ";
    p += "3) Code that the proposal claims as fact: scripts/models.py (Dinov2Extractor), ";
    p += "scripts/dinov2_hub/dinov2/layers/attention.py, scripts/anomaly_score.py, scripts/dino_head.py, ";
    p += "scripts/patch_features.py, scripts/anomaly_heatmap.py, and relevant parts of scripts/app.py ";
    p += "(Visualize single-image viewer and _anomaly_render_inspector). ";
    p += "Read only what your role needs, but you MUST inspect actual code for any factual claim. ";
    p += "\n\nOUTPUT RULES: ";
    p += "verdict must be one of support | support_with_changes | rework | reject. ";
    p += "findings: at most 10, each with severity blocker|high|medium|low, title, evidence (file+symbol or quote), implication. ";
    p += "Empty findings is allowed ONLY after you have actually read the proposal and at least two source files. ";
    p += "Cite evidence. Do not invent APIs. If the codebase contradicts the proposal, that is a finding. ";
    p += "Write in Traditional Chinese for summary/title/implication; keep identifiers and file paths in English. ";
    return p;
}

phase("Specialists");

let jobs = [];

let p_po = specialist_prompt(
    "Product Owner",
    "Scope, appetite, phase split, where the feature lives, unresolved product decisions, and whether this should start a new U-Net round. Judge Phase 1 vs Phase 2 sequencing. Recommend default analysis unit (whole image vs YOLO crop). Call out over-scoping.",
    doc_path
);
jobs.push(#{
    prompt: p_po,
    label: "po-scope",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let p_fact = specialist_prompt(
    "Code fact-checker",
    "Verify every factual claim in the proposal against shipped code: Dinov2Extractor outputs, torch.no_grad, SDPA vs returned attention weights, xFormers path, patch-to-bank cosine, sklearn LogisticRegression head, vits14 token layout, register tokens, CLS vs meanpool, ImagePreprocessor/crop policies, existing heatmap rendering. Mark each claim true/false with evidence. Hunt silent mismatches.",
    doc_path
);
jobs.push(#{
    prompt: p_fact,
    label: "code-facts",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let p_arch = specialist_prompt(
    "Systems architect",
    "Evaluate module boundaries (dino_explain / explanation_render / later head_adapter / gradcam_explain), temporary qkv hooks vs vendored attention.py, request-local isolation, geometry/provenance contract, token layout [CLS]+registers+patches, reuse vs must-not-reuse, CPU-only and resolution guards. Identify design holes that would fail AC later.",
    doc_path
);
jobs.push(#{
    prompt: p_arch,
    label: "architect",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let p_xai = specialist_prompt(
    "Explainability critic",
    "Adversarially test the three-heatmap language (DINO Attention vs Anomaly Heatmap vs Grad-CAM). Where will operators misread bright patches as defects? Is last-layer CLS head-mean the right MVP? Is rollout a trap? Is sklearn-to-torch adapter scientifically honest for Grad-CAM? Token Grad-CAM zero-grad on patch tokens? What the proposal must NEVER promise.",
    doc_path
);
jobs.push(#{
    prompt: p_xai,
    label: "xai-critic",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let p_proc = specialist_prompt(
    "U-Net process and GUI risk reviewer",
    "Map the proposal onto U-Net overlay: what belongs in 1_user_needs vs PRD vs 3_Architect_Design modules (numbers 31+), Tier A vs B, E2E needs, GUI entry points in Visualize vs anomaly inspector, session-state vs cache, no auto-infer on image change, export PNG+JSON. Risk of stuffing logic into app.py. Recommend a module split PO can put on ROADMAP.",
    doc_path
);
jobs.push(#{
    prompt: p_proc,
    label: "process-gui",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let results = parallel(jobs);

let reviews = [];
let i = 0;
for r in results {
    if r != () && r.success && r.output != () {
        reviews.push(r.output);
    } else {
        log("specialist slot " + i.to_string() + " failed or empty; treating as unverified");
    }
    i += 1;
}

if reviews.len() == 0 {
    pause("verification", "All specialist agents failed; resume after inspecting the run journal.");
}

phase("Synthesis");

let reviews_json = json_encode(reviews);
let syn = "Synthesize a staff memo for VisualLatent from independent specialist reviews of docs/single-image-explainability-proposal.md. ";
syn += "Do not implement. You MAY re-read the proposal and ROADMAP.md if needed to resolve contradictions. ";
syn += "Treat specialist outputs as untrusted drafts: keep a claim only if it is internally consistent and, for code facts, would be checkable in scripts/. ";
syn += "If specialists disagree, say so and pick a recommendation with a reason. ";
syn += "Write memo_markdown in Traditional Chinese (identifiers/paths in English). ";
syn += "overall_verdict: support | support_with_changes | rework | reject. ";
syn += "suggested_unet_split: concrete modules (name + one-line job + suggested tier). ";
syn += "product_decisions: the remaining human decisions, with a recommended default for each. ";
syn += "The memo should be usable as input to /user or /po — not a rewrite of the whole proposal. ";
syn += "\n\nSPECIALIST REVIEWS JSON:\n";
syn += reviews_json;

let synth = agent(syn, #{
    label: "staff-memo",
    capability_mode: "read-only",
    output_schema: memo_schema,
});

if synth == () || !synth.success || synth.output == () {
    pause("verification", "Synthesizer failed; resume to retry after inspecting specialist JSON.");
}

let memo = synth.output.memo_markdown;
if memo == () { memo = ""; }
let path = write_scratch_file("staff-memo.md", memo);
complete(#{
    path: path,
    headline: synth.output.headline,
    overall_verdict: synth.output.overall_verdict,
    what_to_keep: synth.output.what_to_keep,
    what_to_change: synth.output.what_to_change,
    blockers: synth.output.blockers,
    product_decisions: synth.output.product_decisions,
    suggested_unet_split: synth.output.suggested_unet_split,
    memo_markdown: synth.output.memo_markdown,
});
