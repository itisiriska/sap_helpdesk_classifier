import json

import numpy as np
import pandas as pd
import panel as pn
import param
import requests

from config import available_models, BASE_URL, model_names_mapping


class DataView(param.Parameterized):

    model_selector = param.ObjectSelector(
        default=available_models[0],
        objects=available_models,
        label='Select model:'
    )
    update_int = param.Integer(0)

    text = pn.widgets.TextAreaInput()
    button_generate = pn.widgets.Button(name='Predict for random text', button_type='primary')
    button_predict = pn.widgets.Button(name='Predict label', button_type='primary')

    predict_url = "/api/svm/predict/"
    text_response = {'': ''}
    text_report = pn.Pane(
        pd.DataFrame.from_dict(
            text_response,
            orient='index'
        ),
    )

    def _on_predict(self, event):
        model = model_names_mapping[self.model_selector]
        self.predict_url = f"/api/{model}/predict/"
        data = {"text": self.text.value}
        print(json.dumps(data))
        print(BASE_URL + self.predict_url)
        self.text_response = requests.post(
            BASE_URL + self.predict_url,
            json=data
        ).json()
        print(self.text_response)
        self.update_int += 1
        return

    def _on_generate(self, event):
        idx = np.random.randint(0, 3863)
        model = model_names_mapping[self.model_selector]
        self.predict_url = f"/api/{model}/predict/{idx}"
        data = {"text": self.text.value}
        print(BASE_URL + self.predict_url)
        self.text_response = requests.post(
            BASE_URL + self.predict_url,
            json=json.dumps(data)
        ).json()
        print(self.text_response)
        self.update_int += 1
        return

    @param.depends('model_selector', 'update_int')
    def view(self):
        self.button_predict.on_click(self._on_predict)
        self.button_generate.on_click(self._on_generate)

        model = model_names_mapping[self.model_selector]
        performance_response = requests.get(BASE_URL + f'/api/{model}/performance').json()
        model_info = performance_response["model_info"]

        model_info_view = pn.Pane(
            pd.DataFrame.from_dict(
                model_info,
                orient='index'
            ),
        )

        text_report = pn.Pane(
            pd.DataFrame.from_dict(
                self.text_response,
                orient='index'
            ),
        )

        main_view = pn.Row(
            pn.Column(
                '# Analyze text',
                self.text,
                self.button_predict,
                self.button_generate,
                sizing_mode='stretch_width'
            ),
            pn.Column(
                '# Text Report',
                text_report,
                '# Model description',
                model_info_view,
                sizing_mode='stretch_width'
            ),
        )
        return main_view


class BootstrapView(param.Parameterized):

    def __init__(self, main_view, **params):
        super().__init__(**params)
        self.main_view = main_view

    def view(self):
        bootstrap_view = pn.template.BootstrapTemplate(title='SAP helpdesk tickets classifier')
        bootstrap_view.main.append(
            self.main_view.param['model_selector'],
        )
        bootstrap_view.main.append(
            self.main_view.view
        )
        return bootstrap_view


if __name__ == '__main__':
    main_view = DataView()
    bootstrap_view = BootstrapView(main_view).view()
    bootstrap_view.show(port=12345)
