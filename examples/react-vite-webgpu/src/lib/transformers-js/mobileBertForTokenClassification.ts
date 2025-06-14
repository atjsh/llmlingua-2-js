import {
  MobileBertPreTrainedModel,
  TokenClassifierOutput
} from "@huggingface/transformers";

export class MobileBertForTokenClassification extends MobileBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs: object): Promise<TokenClassifierOutput> {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
}
