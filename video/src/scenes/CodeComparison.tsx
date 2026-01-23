import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { CodeBlock } from "../components/CodeBlock";
import { ComparisonBadge } from "../components/ComparisonBadge";
import { colors, gradients } from "../styles/colors";
import { fonts, typography } from "../styles/fonts";

interface CodeComparisonProps {
  title: string;
  beforeCode: string;
  afterCode: string;
  badge: string;
  beforeLabel?: string;
  afterLabel?: string;
  transitionStart?: number;
}

export const CodeComparison: React.FC<CodeComparisonProps> = ({
  title,
  beforeCode,
  afterCode,
  badge,
  beforeLabel = "Before",
  afterLabel = "After (sklab)",
  transitionStart = 200,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  const titleScale = spring({
    frame,
    fps,
    config: { damping: 200 },
    from: 0.95,
    to: 1,
  });

  // Transition animation with easing
  const isTransitioning = frame >= transitionStart;
  const transitionDuration = 40;

  const transitionProgress = interpolate(
    frame,
    [transitionStart, transitionStart + transitionDuration],
    [0, 1],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.inOut(Easing.quad),
    }
  );

  // Before code animations
  const beforeOpacity = interpolate(transitionProgress, [0, 0.4], [1, 0], {
    extrapolateRight: "clamp",
  });
  const beforeScale = interpolate(transitionProgress, [0, 0.5], [1, 0.95], {
    extrapolateRight: "clamp",
  });
  const beforeBlur = interpolate(transitionProgress, [0, 0.4], [0, 8], {
    extrapolateRight: "clamp",
  });

  // After code animations
  const afterOpacity = interpolate(transitionProgress, [0.3, 0.8], [0, 1], {
    extrapolateLeft: "clamp",
  });
  const afterScale = interpolate(transitionProgress, [0.3, 1], [1.02, 1], {
    extrapolateLeft: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      {/* Background glow that changes color */}
      <div
        style={{
          position: "absolute",
          top: "40%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 800,
          height: 400,
          background: isTransitioning
            ? gradients.radialGlowGreen
            : "radial-gradient(ellipse 60% 40% at 50% 50%, rgba(239, 68, 68, 0.08) 0%, transparent 60%)",
          opacity: interpolate(transitionProgress, [0, 1], [0.5, 0.8]),
          transition: "background 0.5s",
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 0,
          background: gradients.subtleVignette,
        }}
      />

      {/* Content */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          padding: "50px 80px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Title */}
        <div
          style={{
            fontFamily: fonts.sans,
            fontSize: typography.title.fontSize,
            fontWeight: 700,
            color: colors.text,
            marginBottom: 32,
            opacity: titleOpacity,
            transform: `scale(${titleScale})`,
            textAlign: "center",
            letterSpacing: typography.title.letterSpacing,
          }}
        >
          {title}
        </div>

        {/* Code blocks container */}
        <div
          style={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            position: "relative",
          }}
        >
          {/* Before code */}
          <div
            style={{
              position: "absolute",
              width: "85%",
              maxWidth: 1500,
              opacity: beforeOpacity,
              transform: `scale(${beforeScale})`,
              filter: `blur(${beforeBlur}px)`,
            }}
          >
            <div
              style={{
                fontFamily: fonts.sans,
                fontSize: 22,
                color: colors.warningRed,
                marginBottom: 16,
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: colors.warningRed,
                }}
              />
              {beforeLabel}
            </div>
            <CodeBlock
              code={beforeCode}
              startFrame={20}
              framesPerChar={1}
              fontSize={22}
              typewriter={!isTransitioning}
            />
          </div>

          {/* After code */}
          {isTransitioning && (
            <div
              style={{
                position: "absolute",
                width: "85%",
                maxWidth: 1500,
                opacity: afterOpacity,
                transform: `scale(${afterScale})`,
              }}
            >
              <div
                style={{
                  fontFamily: fonts.sans,
                  fontSize: 22,
                  color: colors.successGreenBright,
                  marginBottom: 16,
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: colors.successGreen,
                    boxShadow: `0 0 8px ${colors.successGreenGlow}`,
                  }}
                />
                {afterLabel}
              </div>
              <CodeBlock
                code={afterCode}
                startFrame={0}
                framesPerChar={2}
                fontSize={22}
                typewriter={true}
                glowColor={colors.successGreenGlow}
              />
            </div>
          )}
        </div>

        {/* Badge */}
        <div
          style={{
            position: "absolute",
            bottom: 60,
            right: 80,
          }}
        >
          <ComparisonBadge text={badge} startFrame={transitionStart + 50} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Pre-configured comparisons
export const CrossValidationComparison: React.FC = () => (
  <CodeComparison
    title="Cross-Validation"
    beforeLabel="sklearn"
    beforeCode={`from sklearn.model_selection import cross_validate

scores = cross_validate(
    pipeline, X, y, cv=5,
    scoring={"accuracy": "accuracy"},
    return_train_score=False
)
mean_acc = scores["test_accuracy"].mean()
std_acc = scores["test_accuracy"].std()`}
    afterCode={`result = experiment.cross_validate(X, y, cv=5)

mean_acc = result.metrics["cv/accuracy_mean"]`}
    badge="5x less code"
    transitionStart={200}
  />
);

export const GridSearchComparison: React.FC = () => (
  <CodeComparison
    title="Grid Search"
    beforeLabel="sklearn"
    beforeCode={`from sklearn.model_selection import GridSearchCV

searcher = GridSearchCV(
    pipeline,
    param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
    scoring="accuracy",
    cv=5,
    refit=True
)
searcher.fit(X, y)
best_params = searcher.best_params_`}
    afterCode={`result = experiment.search(
    GridSearchConfig(
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]}
    ),
    X, y, cv=5
)`}
    badge="3x less code"
    transitionStart={200}
  />
);

export const OptunaSearchComparison: React.FC = () => (
  <CodeComparison
    title="Optuna Search"
    beforeLabel="optuna + sklearn"
    beforeCode={`import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "model__C": trial.suggest_float("C", 1e-3, 1e2, log=True)
    }
    estimator = clone(pipeline).set_params(**params)
    return cross_val_score(estimator, X, y, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)`}
    afterCode={`result = experiment.search(
    OptunaConfig(search_space=search_space, n_trials=50),
    X, y, cv=5
)`}
    badge="4x less code"
    transitionStart={250}
  />
);
