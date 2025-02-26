import unittest
import json
from app import create_app  # Import your Flask app factory

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()  # Create a test client
        self.app.testing = True  # Enable testing mode

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'healthy')

    def test_validate_columns(self):
        with self.app.test_request_context(data={'columns1[]': 'address', 'columns2[]': 'city'}):
            from app import validate_columns  # Import inside the test to avoid circular imports
            columns1, columns2 = validate_columns(request.form)
            self.assertEqual(columns1, ['address'])
            self.assertEqual(columns2, ['city'])

    def test_validate_columns_missing(self):
        with self.assertRaises(ValueError):
            with self.app.test_request_context():
                from app import validate_columns
                validate_columns(request.form)

if __name__ == '__main__':
    unittest.main()