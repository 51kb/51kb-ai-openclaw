import type { BootstrapBudgetAnalysis } from "./bootstrap-budget.js";
import type { ThinkLevel } from "../auto-reply/thinking.js";
import { THINKING_LEVEL_RANKS } from "../auto-reply/thinking.shared.js";

/**
 * Signals used to assess task complexity for adaptive thinking level decisions.
 * All values arecheap to compute at the point where they are collected.
 */
export type AdaptiveThinkingSignals = {
  /** Characters in the latest user prompt */
  promptChars: number;
  /** Total messages in the session so far */
  messageCount: number;
  /** Assistant turns in the session history */
  assistantTurnCount: number;
  /** Number of available tools exposed to the agent */
  toolCount: number;
  /** Bootstrap truncation analysis (undefined when bootstrapping is disabled) */
  bootstrapAnalysis?: BootstrapBudgetAnalysis;
  /** True when the session has prior tool-error history */
  hasErrorRecovery: boolean;
  /** Session age in rounds (incremental counter maintained by the runner) */
  sessionRound: number;
};

type ComplexityLevel = "trivial" | "simple" | "moderate" | "complex" | "intricate";

const COMPLEXITY_THRESHOLDS = {
  trivial: 0,
  simple: 15,
  moderate: 40,
  complex: 70,
  // > 70 = intricate
} as const;

const SESSION_AGE_THRESHOLDS = {
  /** New conversation — fresh context, no history penalty */
  fresh: 0,
  /** Light history — 1–5 rounds in */
  warm: 6,
  /** Moderate history — 6–15 rounds */
  loaded: 16,
  /** Heavy history — > 15 rounds */
  heavy: 30,
} as const;

/**
 * Classify the session's history burden.
 * Heavy history makes it harder for the model to stay focused; we compensate
 * by asking for more explicit reasoning.
 */
function classifySessionAge(sessionRound: number): keyof typeof SESSION_AGE_THRESHOLDS {
  if (sessionRound <= SESSION_AGE_THRESHOLDS.fresh) return "fresh";
  if (sessionRound <= SESSION_AGE_THRESHOLDS.warm) return "warm";
  if (sessionRound <= SESSION_AGE_THRESHOLDS.loaded) return "loaded";
  return "heavy";
}

/**
 * Score a single complexity signal on a 0–100 scale.
 * Returns [baseScore, weight] so callers can compute a weighted average.
 */
function scorePromptChars(chars: number): [number, number] {
  // 0–500 chars = trivial (score 0)
  // 500–2 000 = simple (0–25)
  // 2 000–8 000 = moderate (25–60)
  // 8 000–25 000 = complex (60–85)
  // > 25 000 = intricate (85–100)
  if (chars <= 500) return [0, 1.0];
  if (chars <= 2000) return [Math.round(((chars - 500) / 1500) * 25), 1.2];
  if (chars <= 8000) return [Math.round(25 + ((chars - 2000) / 6000) * 35), 1.5];
  if (chars <= 25000) return [Math.round(60 + ((chars - 8000) / 17000) * 25), 1.8];
  return [Math.min(100, 85 + ((chars - 25000) / 25000) * 15), 2.0];
}

function scoreMessageCount(count: number): [number, number] {
  // 0–3 msgs = trivial, 4–15 = simple, 16–40 = moderate, >40 = complex
  if (count <= 3) return [0, 0.8];
  if (count <= 15) return [Math.round(((count - 3) / 12) * 25), 1.0];
  if (count <= 40) return [Math.round(25 + ((count - 15) / 25) * 35), 1.3];
  return [Math.min(100, 60 + ((count - 40) / 20) * 20), 1.5];
}

function scoreToolCount(count: number): [number, number] {
  // > 50 tools = significant choice burden
  if (count <= 20) return [0, 0.5];
  if (count <= 50) return [Math.round(((count - 20) / 30) * 30), 0.8];
  if (count <= 100) return [Math.round(30 + ((count - 50) / 50) * 30), 1.2];
  return [Math.min(100, 60 + ((count - 100) / 50) * 20), 1.5];
}

function scoreBootstrapAnalysis(analysis: BootstrapBudgetAnalysis | undefined): [number, number] {
  if (!analysis) return [0, 0];
  const { hasTruncation, truncatedFiles, nearLimitFiles } = analysis;
  if (!hasTruncation && truncatedFiles.length === 0) return [0, 0.5];
  // Bootstrap truncation means the agent has partial context — harder to reason correctly
  const truncationScore = Math.min(100, truncatedFiles.length * 15 + nearLimitFiles.length * 5);
  return [truncationScore, 1.5];
}

/**
 * Compute a weighted composite complexity score (0–100) from the full signal bag.
 *
 * The score reflects how much "thinking work" the model needs to do, not how
 * smart the model is.  A high score means the task has many moving parts, deep
 * history, or significant context gaps — all signals that benefit from deeper
 * reasoning.
 */
export function computeComplexityScore(signals: AdaptiveThinkingSignals): number {
  const sessionAge = classifySessionAge(signals.sessionRound);
  const ageScores: Record<keyof typeof SESSION_AGE_THRESHOLDS, [number, number]> = {
    fresh: [0, 0.2],
    warm: [10, 0.5],
    loaded: [30, 0.8],
    heavy: [55, 1.2],
  };
  const [ageScore, ageWeight] = ageScores[sessionAge];

  const [promptScore, promptWeight] = scorePromptChars(signals.promptChars);
  const [msgScore, msgWeight] = scoreMessageCount(signals.messageCount);
  const [toolScore, toolWeight] = scoreToolCount(signals.toolCount);
  const [bootScore, bootWeight] = scoreBootstrapAnalysis(signals.bootstrapAnalysis);

  // Error recovery adds a one-time complexity spike
  const errorPenalty = signals.hasErrorRecovery ? 20 : 0;

  const totalWeight = promptWeight + msgWeight + toolWeight + bootWeight + ageWeight;
  if (totalWeight === 0) return 0;

  const weighted =
    (promptScore * promptWeight +
      msgScore * msgWeight +
      toolScore * toolWeight +
      bootScore * bootWeight +
      ageScore * ageWeight) /
    totalWeight;

  return Math.min(100, Math.round(weighted + errorPenalty));
}

/**
 * Classify a numeric score into a named complexity tier.
 */
export function classifyComplexity(score: number): ComplexityLevel {
  if (score < COMPLEXITY_THRESHOLDS.simple) return "trivial";
  if (score < COMPLEXITY_THRESHOLDS.moderate) return "simple";
  if (score < COMPLEXITY_THRESHOLDS.complex) return "moderate";
  if (score < 100) return "complex";
  return "intricate";
}

/**
 * Determine a ThinkLevel adjustment given the computed complexity tier and the
 * model-capability-adjusted base level.
 *
 * Adjustment strategy
 * ─────────────────────
 * - **trivial / simple** : downgrade one step (save latency + cost)
 * - **moderate**          : use the base level as-is (already appropriate)
 * - **complex**           : upgrade one step (deep reasoning helps)
 * - **intricate**         : upgrade two steps, capped at the model's maximum
 *
 * The adjustment is applied relative to the *base* level so that upgrades
 * and downgrades are symmetric regardless of where the default lands.
 */
export function resolveThinkLevelAdjustment(
  baseLevel: ThinkLevel,
  complexity: ComplexityLevel,
  modelMaxLevel: ThinkLevel,
): ThinkLevel {
  const rank = (level: ThinkLevel): number => THINKING_LEVEL_RANKS[level] ?? 0;
  const clamp = (level: ThinkLevel): ThinkLevel => {
    if (rank(level) > rank(modelMaxLevel)) return modelMaxLevel;
    return level;
  };

  const levels: ThinkLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
  const baseIdx = levels.indexOf(baseLevel);
  if (baseIdx === -1) return baseLevel;

  switch (complexity) {
    case "trivial":
    case "simple":
      // downgrade one step (but never below "off")
      return clamp(levels[Math.max(0, baseIdx - 1)]);
    case "moderate":
      return baseLevel;
    case "complex":
      // upgrade one step
      return clamp(levels[Math.min(levels.length - 1, baseIdx + 1)]);
    case "intricate":
      // upgrade two steps
      return clamp(levels[Math.min(levels.length - 1, baseIdx + 2)]);
  }
}

/**
 * Full adaptive thinking level resolver.
 *
 * @param signals       — current session signals
 * @param baseLevel     — model-capability-adjusted default (e.g. from `model-thinking-default.ts`)
 * @param modelMaxLevel — highest thinking level the model supports
 */
export function resolveAdaptiveThinkLevel(
  signals: AdaptiveThinkingSignals,
  baseLevel: ThinkLevel,
  modelMaxLevel: ThinkLevel,
): ThinkLevel {
  const score = computeComplexityScore(signals);
  const complexity = classifyComplexity(score);
  return resolveThinkLevelAdjustment(baseLevel, complexity, modelMaxLevel);
}
