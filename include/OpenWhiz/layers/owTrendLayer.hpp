/*
 * owTrendLayer.hpp
 *
 *  Created on: Apr 12, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <vector>
#include <algorithm>

namespace ow {

/**
 * @brief A layer that analyzes and amplifies temporal trends in windowed data.
 * It calculates the slope (Last - First) and applies a learnable boost to the 
 * most recent signal, helping the model capture momentum.
 */
class owTrendLayer : public owLayer {
public:
    owTrendLayer(size_t inputSize = 0) : m_size(inputSize), m_params(1, 1), m_grads(1, 1) {
        m_layerName = "Trend Layer";
        m_params(0, 0) = 0.01f; // Safe initial trend sensitivity
    }

    size_t getInputSize() const override { return m_size; }
    size_t getOutputSize() const override { return m_size; }
    void setNeuronNum(size_t num) override { m_size = num; }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        owTensor<float, 2> output = input;
        
        size_t batch = input.shape()[0];
        size_t win = input.shape()[1];
        
        if (win < 2) return output;

        // Apply clamped scale to prevent explosion
        float scale = std::max(0.0f, std::min(0.5f, m_params(0, 0)));
        for (size_t i = 0; i < batch; ++i) {
            float slope = input(i, win - 1) - input(i, 0);
            output(i, win - 1) += scale * slope;
        }
        
        return output;
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGrad = outputGradient;
        
        size_t batch = m_lastInput.shape()[0];
        size_t win = m_lastInput.shape()[1];
        float scale = std::max(0.0f, std::min(0.5f, m_params(0, 0)));
        
        float dScale = 0;
        for (size_t i = 0; i < batch; ++i) {
            float slope = m_lastInput(i, win - 1) - m_lastInput(i, 0);
            dScale += outputGradient(i, win - 1) * slope;
            
            inputGrad(i, win - 1) *= (1.0f + scale);
            inputGrad(i, 0) -= outputGradient(i, win - 1) * scale;
        }
        
        // Manual gradient clipping for the scale parameter
        if (dScale > 1.0f) dScale = 1.0f;
        if (dScale < -1.0f) dScale = -1.0f;

        m_grads(0, 0) += dScale;
        return inputGrad;
    }

    void train() override {
        if (m_optimizer && !m_isFrozen) {
            m_optimizer->update(m_params, m_grads);
            // Force parameter back into safe range after update
            m_params(0, 0) = std::max(0.0f, std::min(0.5f, m_params(0, 0)));
        }
        m_grads.setZero();
    }

    float* getParamsPtr() override { return m_params.data(); }
    float* getGradsPtr() override { return m_grads.data(); }
    size_t getParamsCount() override { return 1; }

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owTrendLayer>(m_size);
        copy->m_params(0, 0) = m_params(0, 0);
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<Size>" << m_size << "</Size>\n";
        ss << "<SlopeScale>" << m_params(0, 0) << "</SlopeScale>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_size = std::stoul(getTagContent(xml, "Size"));
        m_params(0, 0) = std::stof(getTagContent(xml, "SlopeScale"));
    }

private:
    size_t m_size;
    owTensor<float, 2> m_params, m_grads;
    owTensor<float, 2> m_lastInput;
};

} // namespace ow
