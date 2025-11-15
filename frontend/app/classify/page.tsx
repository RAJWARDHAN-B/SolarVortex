"use client";

import { useState } from "react";
import ImageUpload from "../components/ImageUpload";
import ClassificationResults from "../components/ClassificationResults";

export interface SolarPanelClassification {
  id: string;
  image: File | string;
  type: string;
  size: string;
  brand?: string;
  model?: string;
  age?: string;
  recyclabilityGrade: string;
  materialComposition?: {
    glass: number;
    aluminum: number;
    silicon: number;
    other: number;
  };
  dimensions?: {
    length: number;
    width: number;
    thickness: number;
  };
  powerRating?: string;
  condition?: string;
}

export default function ClassifyPage() {
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [classifications, setClassifications] = useState<SolarPanelClassification[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImagesUpload = (files: File[]) => {
    setUploadedImages((prev) => [...prev, ...files]);
  };

  const handleRemoveImage = (index: number) => {
    setUploadedImages((prev) => prev.filter((_, i) => i !== index));
    setClassifications((prev) => prev.filter((_, i) => i !== index));
  };

  const handleClassify = async () => {
    if (uploadedImages.length === 0) return;

    setIsProcessing(true);
    
    // Simulate API call - Replace with actual API endpoint
    try {
      // TODO: Replace with actual API call
      // const formData = new FormData();
      // uploadedImages.forEach((file) => {
      //   formData.append('images', file);
      // });
      // const response = await fetch('/api/classify', {
      //   method: 'POST',
      //   body: formData
      // });
      // const data = await response.json();

      // Simulated classification results
      await new Promise((resolve) => setTimeout(resolve, 2000));
      
      const mockClassifications: SolarPanelClassification[] = uploadedImages.map((file, index) => ({
        id: `panel-${Date.now()}-${index}`,
        image: file,
        type: ["Monocrystalline", "Polycrystalline", "Thin-Film"][index % 3],
        size: ["Standard (1.6m x 1m)", "Large (2m x 1m)", "Small (1.2m x 0.8m)"][index % 3],
        brand: ["SunPower", "LG Solar", "Canadian Solar"][index % 3],
        model: `SP-${Math.floor(Math.random() * 1000)}`,
        age: `${Math.floor(Math.random() * 15) + 5} years`,
        recyclabilityGrade: ["A", "B", "C"][index % 3],
        materialComposition: {
          glass: 70 + Math.random() * 10,
          aluminum: 10 + Math.random() * 5,
          silicon: 5 + Math.random() * 3,
          other: 15 - Math.random() * 5,
        },
        dimensions: {
          length: [1600, 2000, 1200][index % 3],
          width: [1000, 1000, 800][index % 3],
          thickness: 40 + Math.random() * 10,
        },
        powerRating: `${250 + Math.floor(Math.random() * 100)}W`,
        condition: ["Good", "Fair", "Poor"][index % 3],
      }));

      setClassifications(mockClassifications);
    } catch (error) {
      console.error("Classification error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Page Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            Solar Panel Classification
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            Upload images to classify and analyze solar panels using AI
          </p>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Upload Solar Panel Images
              </h2>
              <ImageUpload
                onImagesUpload={handleImagesUpload}
                uploadedImages={uploadedImages}
                onRemoveImage={handleRemoveImage}
              />
              
              {uploadedImages.length > 0 && (
                <button
                  onClick={handleClassify}
                  disabled={isProcessing}
                  className="w-full mt-6 bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Classify Panels
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {classifications.length > 0 ? (
              <ClassificationResults classifications={classifications} />
            ) : (
              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
                <div className="text-center py-12">
                  <svg className="mx-auto h-16 w-16 text-slate-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <h3 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">
                    Classification Results
                  </h3>
                  <p className="text-slate-500 dark:text-slate-400">
                    Upload images and click "Classify Panels" to see results
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

