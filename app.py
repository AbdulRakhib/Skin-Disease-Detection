from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
from matplotlib import pyplot as plt

app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username", "class": "input--style-4"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password", "class": "input--style-4"})
    submit = SubmitField('Register', render_kw={"class": "btn btn--radius-2 btn--blue"})

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username", "class": "input--style-4"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password", "class": "input--style-4"})
    submit = SubmitField('Login', render_kw={"class": "btn btn--radius-2 btn--blue"})


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    # Retrieve the user from the database
    user = User.query.filter_by(id=current_user.id).first()

    return render_template('dashboard.html', username=user.username.upper())

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/imageUpload', methods=['POST'])
def imageUpload():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return 'No image file was uploaded.'
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No image file was selected.'

    imageFile = request.files['image']
    img = Image.open(imageFile)
    f = open('modelEnsemble.pkl', 'rb')
    loadedModel = pickle.load(f)

    plt.imshow(img)

    image = img.convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image, dtype=np.float32)
    image /= 255.0
    image = np.expand_dims(image, 0)
    input_tensor = tf.convert_to_tensor(image)

    prediction = loadedModel.predict(input_tensor)
    classNames = ['Acne', 'Eczema', 'Melanoma Skin Cancer Nevi and Moles', 'Psoriasis']
    predicted_class = classNames[np.argmax(prediction)]
    confidence = round(100 * (np.max(prediction)))
    plt.title(f"Predicted Disease: {predicted_class}.\n Accuracy level: {confidence}%")
    plt.axis("off")
    plt.show()

    f.close()

    user = User.query.filter_by(id=current_user.id).first()
    return render_template('dashboard.html', username=user.username.upper())



if __name__ == "__main__":
    #app.run(host="0.0.0.0", debug=True)
    app.run(debug=True)
