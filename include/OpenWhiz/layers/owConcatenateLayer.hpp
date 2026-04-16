/*
 * owConcatenateLayer.hpp
 *
 *  Created on: Apr 10, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <vector>
#include <memory>
#include <numeric>

namespace ow {

/**
 * @class owConcatenateLayer
 * @brief A structural meta-layer that implements the "Independent Experts" architecture by parallelizing network branches.
 * 
 * This layer acts as a powerful orchestration point, allowing multiple independent neural network 
 * pathways (Experts) to process the same or sliced input data. The results are merged into 
 * a single wide feature vector for final decision-making.
 * 
 * **Architectural Significance:**
 * - **Multi-Scale Analysis:** Allows implementing branches that look at different temporal 
 *   resolutions (e.g., Weekly vs. Monthly trends in Financial Systems).
 * - **Modular Design:** Experts can be complex sequences of layers, including LSTMs or 
 *   PCA-enhanced blocks.
 * 
 * **Mathematical Logic:**
 * - **Forward Pass:** 
 *   Let $X$ be the input. The output is a horizontal concatenation:
 *   $Y = [Expert_1(X), Expert_2(X), ..., Expert_N(X)]$
 *   Total Output Width = $\sum OutputSize_i$.
 * - **Backward Pass (Gradient Flow):**
 *   The upstream gradient $\delta$ is sliced into $\{\delta_1, \delta_2, ..., \delta_N\}$. 
 *   If `m_useSharedInput` is enabled, the resulting input gradient is the element-wise sum:
 *   $\nabla X = \sum_{i=1}^{N} \nabla Expert_i(\delta_i)$
 */
class owConcatenateLayer : public owLayer {
public:
    /**
     * @brief Constructs an owConcatenateLayer with an optional initial set of branches.
     * @param branches A vector of shared pointers to the layers that will form the parallel branches.
     * @param useSharedInput If true, all branches receive the full input instead of horizontal slices.
     */
    owConcatenateLayer(const std::vector<std::shared_ptr<owLayer>>& branches = {}, bool useSharedInput = false)
        : m_branches(branches), m_useSharedInput(useSharedInput) {
        m_layerName = "Concatenate Layer";
        m_branchEnabled.assign(m_branches.size(), true);
    }

    /**
     * @brief Returns the list of internal branches.
     */
    std::vector<std::shared_ptr<owLayer>>& getBranches() { return m_branches; }

    /**
     * @brief Enables or disables a specific branch.
     * @param branchNo The index of the branch.
     * @param enable True to enable, false to disable (bypass).
     */
    void enableBranch(int branchNo, bool enable) {
        if (branchNo >= 0 && (size_t)branchNo < m_branchEnabled.size()) {
            m_branchEnabled[branchNo] = enable;
        }
    }

    /**
     * @brief Checks if a specific branch is enabled.
     * @param branchNo The index of the branch.
     * @return True if enabled.
     */
    bool isBranchEnabled(int branchNo) const {
        if (branchNo >= 0 && (size_t)branchNo < m_branchEnabled.size()) {
            return m_branchEnabled[branchNo];
        }
        return false;
    }

    /**
     * @brief Sets whether all branches should receive the full input (Shared) 
     * or a horizontal slice (Standard).
     * @param shared True for shared input, false for sliced input.
     */
    void setUseSharedInput(bool shared) { m_useSharedInput = shared; }

    /**
     * @brief Adds a new parallel branch to the layer.
     * @param branch Shared pointer to the layer to be added as a branch.
     */
    void addBranch(std::shared_ptr<owLayer> branch) {
        if (branch) {
            m_branches.push_back(branch);
            m_branchEnabled.push_back(true);
        }
    }

    /**
     * @brief Replaces the current branches with a new set.
     * @param branches Vector of shared pointers to the new layers.
     */
    void setBranches(const std::vector<std::shared_ptr<owLayer>>& branches) {
        m_branches = branches;
        m_branchEnabled.assign(m_branches.size(), true);
    }

    /**
     * @brief Calculates the total input size expected by this layer, considering only enabled branches.
     * @return The sum of the input sizes of all active branches.
     */
    size_t getInputSize() const override {
        if (m_useSharedInput) {
            for (size_t i = 0; i < m_branches.size(); ++i) {
                if (m_branchEnabled[i]) return m_branches[i]->getInputSize();
            }
            return 0;
        }
        size_t total = 0;
        for (size_t i = 0; i < m_branches.size(); ++i) {
            if (m_branchEnabled[i]) total += m_branches[i]->getInputSize();
        }
        return total;
    }

    /**
     * @brief Calculates the total output size produced by this layer, considering only enabled branches.
     * @return The sum of the output sizes of all active branches.
     */
    size_t getOutputSize() const override {
        size_t total = 0;
        for (size_t i = 0; i < m_branches.size(); ++i) {
            if (m_branchEnabled[i]) total += m_branches[i]->getOutputSize();
        }
        return total;
    }

    /**
     * @brief Implementation of virtual setNeuronNum. 
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Performs the forward pass, skipping disabled branches.
     * @param input Input tensor of shape [BatchSize, TotalInputSize].
     * @return Concatenated output tensor from active branches.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batch = input.shape()[0];
        m_outputs.clear();
        m_outputs.reserve(m_branches.size());
        m_activeOutputIndices.clear();

        size_t currentInOffset = 0;
        for (size_t k = 0; k < m_branches.size(); ++k) {
            if (!m_branchEnabled[k]) {
                if (!m_useSharedInput) currentInOffset += m_branches[k]->getInputSize();
                continue;
            }

            m_activeOutputIndices.push_back(k);
            if (m_useSharedInput) {
                m_outputs.push_back(m_branches[k]->forward(input));
            } else {
                size_t inSize = m_branches[k]->getInputSize();
                owTensor<float, 2> slicedInput(batch, inSize);
                for (size_t i = 0; i < batch; ++i) {
                    for (size_t j = 0; j < inSize; ++j) {
                        slicedInput(i, j) = input(i, currentInOffset + j);
                    }
                }
                m_outputs.push_back(m_branches[k]->forward(slicedInput));
                currentInOffset += inSize;
            }
        }

        owTensor<float, 2> result(batch, getOutputSize());
        size_t currentOutOffset = 0;
        for (const auto& out : m_outputs) {
            size_t outSize = out.shape()[1];
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < outSize; ++j) {
                    result(i, currentOutOffset + j) = out(i, j);
                }
            }
            currentOutOffset += outSize;
        }

        return result;
    }

    /**
     * @brief Performs the backward pass, routing gradients only to active branches.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = outputGradient.shape()[0];
        std::vector<owTensor<float, 2>> activeInputGradients;
        activeInputGradients.reserve(m_outputs.size());

        size_t currentOutOffset = 0;
        for (size_t k = 0; k < m_outputs.size(); ++k) {
            size_t outSize = m_outputs[k].shape()[1];
            
            owTensor<float, 2> slicedGrad(batch, outSize);
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < outSize; ++j) {
                    slicedGrad(i, j) = outputGradient(i, currentOutOffset + j);
                }
            }
            
            size_t branchIdx = m_activeOutputIndices[k];
            activeInputGradients.push_back(m_branches[branchIdx]->backward(slicedGrad));
            currentOutOffset += outSize;
        }

        if (m_useSharedInput) {
            if (activeInputGradients.empty()) return owTensor<float, 2>(batch, getInputSize());
            
            size_t inSize = activeInputGradients[0].shape()[1];
            owTensor<float, 2> result(batch, inSize);
            result.setZero();
            for (const auto& grad : activeInputGradients) {
                result = result + grad;
            }
            return result;
        } else {
            owTensor<float, 2> result(batch, getInputSize());
            size_t currentInOffset = 0;
            for (const auto& inGrad : activeInputGradients) {
                size_t inSize = inGrad.shape()[1];
                for (size_t i = 0; i < batch; ++i) {
                    for (size_t j = 0; j < inSize; ++j) {
                        result(i, currentInOffset + j) = inGrad(i, j);
                    }
                }
                currentInOffset += inSize;
            }
            return result;
        }
    }

    /**
     * @brief Triggers training for active branches.
     */
    void train() override {
        if (m_isFrozen) return;
        for (size_t i = 0; i < m_branches.size(); ++i) {
            if (m_branchEnabled[i] && !m_branches[i]->isFrozen()) {
                m_branches[i]->train();
            }
        }
    }

    /**
     * @brief Assigns optimizer to all branches.
     */
    void setOptimizer(owOptimizer* opt) override {
        owLayer::setOptimizer(opt);
        for (auto& branch : m_branches) branch->setOptimizer(opt);
    }

    /**
     * @brief Resets only active branches.
     */
    void reset() override {
        for (size_t i = 0; i < m_branches.size(); ++i) {
            if (m_branchEnabled[i]) m_branches[i]->reset();
        }
    }

    /**
     * @brief Clones the layer and its enabled/disabled state.
     */
    std::shared_ptr<owLayer> clone() const override {
        std::vector<std::shared_ptr<owLayer>> branchCopies;
        for (const auto& b : m_branches) branchCopies.push_back(b->clone());
        auto copy = std::make_shared<owConcatenateLayer>(branchCopies, m_useSharedInput);
        copy->m_layerName = m_layerName;
        copy->m_branchEnabled = m_branchEnabled;
        return copy;
    }

    /**
     * @brief Serializes the layer and branch states to XML.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<BranchCount>" << m_branches.size() << "</BranchCount>\n";
        ss << "<UseSharedInput>" << (m_useSharedInput ? 1 : 0) << "</UseSharedInput>\n";
        for (size_t i = 0; i < m_branches.size(); ++i) {
            ss << "<Branch_" << i << " type=\"" << m_branches[i]->getLayerName() << "\" enabled=\"" << (m_branchEnabled[i] ? 1 : 0) << "\">\n" 
               << m_branches[i]->toXML() << "</Branch_" << i << ">\n";
        }
        return ss.str();
    }

    /**
     * @brief Deserializes parameters and enabled states.
     */
    void fromXML(const std::string& xml) override {
        std::string sharedVal = owLayer::getTagContent(xml, "UseSharedInput");
        if (!sharedVal.empty()) m_useSharedInput = (std::stoi(sharedVal) == 1);
        
        for (size_t i = 0; i < m_branches.size(); ++i) {
            std::string tag = "Branch_" + std::to_string(i);
            std::string branchTagStart = "<" + tag;
            size_t tagPos = xml.find(branchTagStart);
            if (tagPos != std::string::npos) {
                size_t tagEnd = xml.find(">", tagPos);
                std::string fullTag = xml.substr(tagPos, tagEnd - tagPos + 1);
                std::string enabledStr = owLayer::getAttr(fullTag, "enabled");
                if (!enabledStr.empty()) m_branchEnabled[i] = (std::stoi(enabledStr) == 1);
            }
            m_branches[i]->fromXML(owLayer::getNestedTagContent(xml, tag));
        }
    }

    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

private:
    std::vector<std::shared_ptr<owLayer>> m_branches;
    std::vector<bool> m_branchEnabled; ///< Enabling/disabling status for each branch.
    std::vector<owTensor<float, 2>> m_outputs;
    std::vector<size_t> m_activeOutputIndices; ///< Indices of branches that were processed in the last forward pass.
    bool m_useSharedInput = false;
};

} // namespace ow
