# hate-speech-framework

## Example

        import hate-speech-framework  as hsf

        model = hsf.load_model(model_name="ACL")

        trained_model = model.train(train_data,labels)

        model.test(trained_model, test)

>>>