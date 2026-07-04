# Model Registry

`Models` is the central registry of every pretrained checkpoint. Members are addressed
as `Models.<Task>.<Backend>.<Enum>.<member>` and their value is the weight filename;
`Models.download_model(...)` fetches weights from Hugging Face on demand. For the full
catalog of available weights and metrics, see the [Model Zoo](../model-zoo.md).

::: physiotrack.Models
