"use client";

import { SolarPanelClassification } from "../page";

interface ClassificationResultsProps {
  classifications: SolarPanelClassification[];
}

export default function ClassificationResults({
  classifications,
}: ClassificationResultsProps) {
  const getGradeColor = (grade: string) => {
    switch (grade) {
      case "A":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200";
      case "B":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      case "C":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200";
      default:
        return "bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-200";
    }
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 mb-6">
        Classification Results ({classifications.length})
      </h2>
      <div className="space-y-6">
        {classifications.map((classification, index) => (
          <div
            key={classification.id}
            className="border border-slate-200 dark:border-slate-700 rounded-lg p-5 hover:shadow-md transition-shadow"
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              {/* Image Preview */}
              <div className="md:col-span-1">
                <div className="relative aspect-square rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900">
                  {typeof classification.image === "string" ? (
                    <img
                      src={classification.image}
                      alt={`Panel ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <img
                      src={URL.createObjectURL(classification.image)}
                      alt={`Panel ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>
              </div>

              {/* Classification Details */}
              <div className="md:col-span-2 space-y-4">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-3">
                    Panel #{index + 1}
                  </h3>
                  
                  {/* Key Parameters Grid */}
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Type</p>
                      <p className="font-medium text-slate-900 dark:text-slate-100">
                        {classification.type}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Size</p>
                      <p className="font-medium text-slate-900 dark:text-slate-100">
                        {classification.size}
                      </p>
                    </div>
                    {classification.brand && (
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Brand</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {classification.brand}
                        </p>
                      </div>
                    )}
                    {classification.model && (
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Model</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {classification.model}
                        </p>
                      </div>
                    )}
                    {classification.age && (
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Age</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {classification.age}
                        </p>
                      </div>
                    )}
                    {classification.powerRating && (
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Power</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {classification.powerRating}
                        </p>
                      </div>
                    )}
                    {classification.condition && (
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Condition</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {classification.condition}
                        </p>
                      </div>
                    )}
                    <div>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                        Recyclability
                      </p>
                      <span
                        className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${getGradeColor(
                          classification.recyclabilityGrade
                        )}`}
                      >
                        Grade {classification.recyclabilityGrade}
                      </span>
                    </div>
                  </div>

                  {/* Dimensions */}
                  {classification.dimensions && (
                    <div className="mb-4 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
                      <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                        Dimensions
                      </p>
                      <div className="flex gap-4 text-sm">
                        <span className="text-slate-700 dark:text-slate-300">
                          L: {classification.dimensions.length}mm
                        </span>
                        <span className="text-slate-700 dark:text-slate-300">
                          W: {classification.dimensions.width}mm
                        </span>
                        <span className="text-slate-700 dark:text-slate-300">
                          T: {classification.dimensions.thickness.toFixed(1)}mm
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Material Composition */}
                  {classification.materialComposition && (
                    <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
                      <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                        Material Composition
                      </p>
                      <div className="space-y-2">
                        <div>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-700 dark:text-slate-300">Glass</span>
                            <span className="text-slate-600 dark:text-slate-400">
                              {classification.materialComposition.glass.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{
                                width: `${classification.materialComposition.glass}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-700 dark:text-slate-300">Aluminum</span>
                            <span className="text-slate-600 dark:text-slate-400">
                              {classification.materialComposition.aluminum.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-emerald-500 h-2 rounded-full"
                              style={{
                                width: `${classification.materialComposition.aluminum}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-700 dark:text-slate-300">Silicon</span>
                            <span className="text-slate-600 dark:text-slate-400">
                              {classification.materialComposition.silicon.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-purple-500 h-2 rounded-full"
                              style={{
                                width: `${classification.materialComposition.silicon}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-700 dark:text-slate-300">Other</span>
                            <span className="text-slate-600 dark:text-slate-400">
                              {classification.materialComposition.other.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-slate-500 h-2 rounded-full"
                              style={{
                                width: `${classification.materialComposition.other}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

